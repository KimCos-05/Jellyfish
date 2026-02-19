import numpy as np
import cupy as cp
import logging
from pathlib import Path

from script.utility.UnitConverter import UnitConverter
from script.utility.config import Config
from script.utility.manager import DeviceManager
from script.core.simulation_generater import SystemState

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KERNELS_ROOT = PROJECT_ROOT / "script" / "cuda_kernel"
CALCULATOR_ROOT = KERNELS_ROOT / "calculator.cu"

class Integrator:
    """
    CUDA 기반 MD 적분기
    1. CPU 데이터를 GPU로 전송
    2. 매 step마다 CUDA kernel 실행
    3. 필요할 때 결과를 CPU로 전송(저장 등 작업 할 때)
    """

    def __init__(self, system: SystemState, config: Config, converter: UnitConverter):
        logging.info("Integrator 초기화 실행(GPU 메모리 할당)")
        
        # 0. GPU 정보
        logging.info(DeviceManager.get_gpu_info())

        # 1. 시뮬레이션 설정값 (상수) 이관
        self.dt = config.simulation.simulation.dt_fs # 단위 변환 안된 값
        self.n_particles = system.n_particles

        self.target_T_star = converter.T_to_reduced(config.simulation.system.Temp_K)

        # 2. GPU로 변수 이관

        self.gpu_pos = cp.asarray(system.pos.ravel(), dtype = cp.float64)
        self.gpu_vel = cp.asarray(system.vel.ravel(), dtype = cp.float64)
        self.gpu_acc = cp.asarray(system.acc.ravel(), dtype = cp.float64)
        self.gpu_force = cp.asarray(system.force.ravel(), dtype = cp.float64)
        self.gpu_mass = cp.asarray(system.mass, dtype = cp.float64)
        self.gpu_box_size = cp.asarray(system.box_size, dtype = cp.float64)
        self.gpu_particles_id = cp.asarray(system.particles_id, dtype = cp.int32)

        # 3. CUDA kernel 로드 및 컴파일
        
        with open(CALCULATOR_ROOT, 'r', encoding = 'utf-8') as f:
            kernel_code = f.read()
        
        # 모듈 컴파일
        self.module = cp.RawModule(code = kernel_code)
        self.kernel_p1 = self.module.get_function('update_p1')
        self.kernel_force = self.module.get_function('compute_force')
        self.kernel_p2 = self.module.get_function('update_p2')

        # 4. GPU 실행 설정
        self.threads_per_block = config.simulation.simulation.GPU_block_size
        self.blocks_per_grid = (self.n_particles + (self.threads_per_block - 1)) // self.threads_per_block

        # 5. cutoff 설정
        self.cutoff_sq = 9 # 3 * sigma의 제곱, sigma = 1로 정규화 되었으므로 9

        # 6. 격자 설정
        cutoff = np.sqrt(self.cutoff_sq)
        box_size_reduced = system.box_size
        self.grid_dim = np.floor(box_size_reduced / cutoff).astype(np.int32)

        self.cell_size = box_size_reduced / self.grid_dim
        self.num_cells = int(np.prod(self.grid_dim))

        logging.info(f'격자 크기: {self.grid_dim}, 개수: {self.num_cells}개 가 설정되었습니다.')

        self.gpu_head = cp.full(self.num_cells, -1, dtype=cp.int32)
        self.gpu_next = cp.full(self.n_particles, -1, dtype=cp.int32)

        self.gpu_grid_dim = cp.asarray(self.grid_dim, dtype=cp.int32)
        self.gpu_cell_size = cp.asarray(self.cell_size, dtype=cp.float64)

        self.kernel_init_head = self.module.get_function('init_head')
        self.kernel_build = self.module.get_function('build_cell_list')

        logging.info(f'CUDA Kernel 로드 완료: Blocks = {self.threads_per_block}, Threads = {self.blocks_per_grid}')

    def update_cpu_state(self, system: SystemState):
        """
        GPU 메모리의 pos, vel, force값을 CPU의 SystemState로 복사.
        """

        pos_flat = cp.asnumpy(self.gpu_pos)
        vel_flat = cp.asnumpy(self.gpu_vel)
        force_flat = cp.asnumpy(self.gpu_force)
        id_flat = cp.asnumpy(self.gpu_particles_id)

        system.pos = pos_flat.reshape((self.n_particles, 3))
        system.vel = vel_flat.reshape((self.n_particles, 3))
        system.force = force_flat.reshape((self.n_particles, 3))
        system.particles_id = id_flat

    def apply_thermostat(self, target_temp: float):
        """온도 조절 함수"""

        v_reshaped = self.gpu_vel.reshape(-1, 3)
        v_sq = cp.sum(v_reshaped**2, axis = 1) # v^2 = vx^2 + vy^2 + vz^2

        ke_per_particle = 0.5 * self.gpu_mass * v_sq
        total_ke = cp.sum(ke_per_particle)

        current_temp = (2.0 * total_ke) / (3.0 * self.n_particles) # E = 1.5NK_BT

        if current_temp < 1e-6:
            factor = 1.0

        else:
            factor = cp.sqrt(target_temp / current_temp)

        self.gpu_vel *= factor
        logging.info(f'{current_temp:.4f}에서 {target_temp:.4f} (reduced value)로의 보정을 수행하였습니다.')

    # 메모리 정렬
    def reorder_particles(self):
        """입자를 격자 ID 순서대로 배열"""
        
        # 1. 계산을 위해 (N, 3) 모양으로 변경. pos_nx3[:, 0]은 모든 입자의 x좌표가 됨
        pos_nx3 = self.gpu_pos.reshape(-1, 3)

        # 2. 격자 인덱스 계산
        ix = cp.floor(pos_nx3[:, 0] / self.cell_size[0]).astype(cp.int32)
        iy = cp.floor(pos_nx3[:, 1] / self.cell_size[1]).astype(cp.int32)
        iz = cp.floor(pos_nx3[:, 2] / self.cell_size[2]).astype(cp.int32)

        # 그리드 범위 벗어나지 않게 클램핑
        ix = cp.clip(ix, 0, self.grid_dim[0] - 1)
        iy = cp.clip(iy, 0, self.grid_dim[1] - 1)
        iz = cp.clip(iz, 0, self.grid_dim[2] - 1)

        # 3. 정렬 기준(Key) 생성
        cell_idx = ix + iy * self.grid_dim[0] + iz * self.grid_dim[0] * self.grid_dim[1]
        sorted_indices = cp.argsort(cell_idx)

        # 4. 데이터 재배치
        
        # (N, 3) 데이터들
        self.gpu_pos = pos_nx3[sorted_indices].ravel() # 섞고 다시 1차원으로 펴기
        self.gpu_vel = self.gpu_vel.reshape(-1, 3)[sorted_indices].ravel()
        self.gpu_force = self.gpu_force.reshape(-1, 3)[sorted_indices].ravel()
        
        # (N, 1) 스칼라 데이터들
        self.gpu_mass = self.gpu_mass[sorted_indices]
        self.gpu_particles_id = self.gpu_particles_id[sorted_indices]

        logging.info(f'메모리 정렬됨')

    def integrate(self, steps: int, converter: UnitConverter, thermostat_on: bool = False):
        """
        Cuda Kernel 가동부
        thermostat_on = True : 온도를 일정하게 유지
        """

        blocks_cells = (self.num_cells + self.threads_per_block - 1) // self.threads_per_block
        
        dt_reduced = converter.dt_to_reduced(self.dt)

        N_int = cp.int32(self.n_particles)
        dt_double = cp.float64(dt_reduced)
        cutoff_double = cp.float64(self.cutoff_sq)

        for s in range(steps):

            # GPU 커널 호출부

            # 한번 이동
            self.kernel_p1((self.blocks_per_grid,), (self.threads_per_block,), (self.gpu_pos, self.gpu_vel, self.gpu_force, self.gpu_mass, dt_double, self.gpu_box_size, N_int))

            # 메모리 정렬
            if s % 100 == 0:
                
                f_check = cp.asnumpy(self.gpu_force).reshape(-1, 3)

                avg_force = np.mean(np.linalg.norm(f_check,axis = 1))
                logging.info(f'{DeviceManager.get_gpu_info()}')
                logging.info(f'평균 힘 = {avg_force:.5f}')

                self.reorder_particles()

                if thermostat_on == True:
                    self.apply_thermostat(self.target_T_star) # 온도 조절 함수

            # 격자 초기화 및 배치
            self.kernel_init_head((blocks_cells,), (self.threads_per_block,), (self.gpu_head, self.num_cells))
                                  
            self.kernel_build((self.blocks_per_grid,), (self.threads_per_block,), 
                              (self.gpu_pos, self.gpu_head, self.gpu_next, 
                               self.gpu_box_size, self.gpu_grid_dim, self.gpu_cell_size, N_int))

            # 힘 계산
            self.kernel_force((self.blocks_per_grid,), (self.threads_per_block,), 
                                   (self.gpu_pos, self.gpu_force, 
                                    self.gpu_head, self.gpu_next, # Linked List 전달
                                    self.gpu_box_size, self.gpu_grid_dim, self.gpu_cell_size, 
                                    cutoff_double, N_int))

            # 나머지 반칸 이동
            self.kernel_p2((self.blocks_per_grid,), (self.threads_per_block,), (self.gpu_vel, self.gpu_force, self.gpu_mass, dt_double, N_int))
        
        cp.cuda.Stream.null.synchronize()