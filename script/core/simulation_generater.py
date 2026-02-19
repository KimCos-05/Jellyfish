import numpy as np
from dataclasses import dataclass
import logging

from script.utility.config import Config
from script.utility.UnitConverter import UnitConverter

@dataclass
class SystemState:
    """
    모든 시뮬레이션 상태를 담는 class
    """

    n_particles: int # 파티클 개수
    box_size: np.ndarray # 공간 크기

    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    force: np.ndarray

    mass: np.ndarray
    type_id: np.ndarray
    particles_id: np.ndarray

    @classmethod
    def initialize_system(cls, config: Config, converter: UnitConverter) -> 'SystemState':
        logging.info("시뮬레이션 초기상태 초기화 시작") 

        # 1. box Size의 단위변환
        L_real = np.array(config.simulation.system.box_size_nm)
        L_reduced = np.array([converter.len_to_reduced(l) for l in L_real])

        logging.info(f'공간 크기 (정규화) : {L_reduced}')

        # 2. 입자 개수 및 종류 확인
        mol_list = list(config.molecules.values())
        if not mol_list:
            raise ValueError('Config에 정의된 분자가 없습니다.')
        
        target_mol = mol_list[0] # 현재 기준으로 list의 첫번쨰 분자를 기준으로 정의

        sim_mol_info = config.simulation.molecules[0]
        n_particles = sim_mol_info.count

        logging.info(f'정규화 기준 입자 : {sim_mol_info.type}, 개수: {n_particles}')

        # 3. 초기 입자 배치

        n_side= int(np.ceil(n_particles**(1/3)))

        dL = L_reduced / n_side

        # ex) np.linspace(0, 10, 5) : [0.0, 2.5, 5.0, 7.5, 10.0]. 0부터 10까지 5등분한다.
        x = np.linspace(dL[0]/2, L_reduced[0] - dL[0]/2, n_side) # dL[0]/2부터 L_reduced[0] - dL[0]/2까지 n_side 등분한다.
        y = np.linspace(dL[1]/2, L_reduced[1] - dL[1]/2, n_side)

        z_center_start = L_reduced[2] * 0.3
        z_center_end = L_reduced[2] * 0.7
        z = np.linspace(z_center_start, z_center_end, n_side)

        # z = np.linspace(dL[2]/2, L_reduced[2] - dL[2]/2, n_side)
        
        # ex) np.meshgrid(x,y,z) : 3차원 좌표계의 모든 배열을 생성. 즉, x = [0,1], y=[0,1]이면 meshgrid하면 (0,0), (0,1), (1,0), (1,1)이 생성된다.
        gx, gy, gz = np.meshgrid(x, y, z, indexing='ij') # indexing = 'xy' : 데카르트 좌표계, indexing = 'ij' : 행렬 좌표계 / 컴퓨터에서 데카르트면 y,x,z순으로 읽음
        gx, gy, gz = gx.ravel(), gy.ravel(), gz.ravel() # 3차원 데이터를 1차원으로 펼치기.

        # axis = 0은 위아래로 쌓기. Shape : (3,N) / axis = 1은 옆으로 쌓기. Shape : (N, 3)
        # x = [1,2,3], y = [4,5,6], z = 7,8,9면 
        # np.stack(x,y,z, axis=0) = [[1,2,3],[4,5,6],[7,8,9]]
        # np.stack(x,y,z, axis=1) = [[1,4,7], [2,5,8], [3,6,9]] -> (x1, y1, z1), (x2, y2, z2), (x3, y3, z1)으로 배열되는 셈.
        pos_lattice = np.stack([gx, gy, gz], axis = 1)

        indices = np.arange(len(pos_lattice)) # 0부터 len(pos_lattice)까지의 숫자를 생성.
        np.random.shuffle(indices)

        # indices로 0부터 len(pos_lattice)까지의 숫자 배열이 생성되어 있음. 이를 shuffle 한 다음, 앞에서부터 n_particles개만큼 뽑는 것.
        # 그러면 남아있는 배열이 index가 되어 중간중간 빈 공간이 생긴 모양이 나옴.
        pos = pos_lattice[indices[:n_particles]]

        # 4. 초기 속도 배치 (볼츠만 분포를 따르도록 배치)

        T_star = converter.T_to_reduced(config.simulation.system.Temp_K)
        vel = np.random.normal(0.0, np.sqrt(T_star), (n_particles, 3)) #정규분포에서 난수 뽑기 / normal(평균, 표준편차, 크기)

        v_cm = np.mean(vel, axis=0)
        vel -= v_cm

        acc = np.zeros((n_particles, 3))
        force = np.zeros((n_particles, 3))

        m_reduced = converter.mass_to_reduced(target_mol.mass_amu)
        mass = np.ones(n_particles) * m_reduced

        type_id = np.zeros(n_particles, dtype=int)

        # 5. 입자 ID 부여 (꼬리표)
        particles_id = np.arange(n_particles, dtype=int)

        logging.info(f'시스템 초기 설정 완료: 입자 수 = {n_particles}, 정규화 온도 = {T_star:.4f}')

        return cls(n_particles = n_particles, 
                   box_size = L_reduced, 
                   pos = pos, 
                   vel = vel, 
                   acc = acc, 
                   force = force, 
                   mass = mass, 
                   type_id = type_id,
                   particles_id = particles_id)