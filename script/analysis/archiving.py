import numpy as np
from pathlib import Path
import logging

from script.core.simulation_generater import SystemState

class Archiving:
    """
    시뮬레이션 결과 저장 class
    """
    def __init__(self, file_path: Path):

        self.filepath = file_path

        with open(self.filepath, 'w') as f:
            pass

        logging.info(f'파일 경로 초기화 완료: {self.filepath}')


    def write_frame(self, step: int, system: SystemState, converter=None, dt_fs = 1.0):
        """
        현재 프레임을 파일에 추가
        Args:
            pos, vel, force: Reduced Unit 상태의 배열 (N,3)
        """

        pos = system.pos
        vel = system.vel
        force = system.force
        ids = system.particles_id

        n_particles = len(pos)

        # 1. 좌표 변환 (Reduced -> Real)

        if converter:
            # 1nm = 10A 이므로 Reduced -> real(nm) -> A 변환
            scale_factor = converter.sigma_ref * 1e9 * 10.0
            pos_real = pos * scale_factor

            scale_vel = converter.sigma_ref / converter.tau
            vel_real = vel * scale_vel

            scale_force = converter.epsilon_ref / converter.sigma_ref
            force_real = force * scale_force

        else:
            pos_real = pos
            vel_real = vel
            force_real = force

        real_time_fs = step * dt_fs

        # 2. xyz 포맷 작성

        with open(self.filepath, 'a') as f:
            f.write(f'{n_particles}\n')
            f.write(f'Step: {step}, Time: {real_time_fs} Properties=species:S:1:pos:R:3:vel:R:3:force:R:3:id:I:1\n')

            for i in range(n_particles):

                x,y,z = pos_real[i]
                vx, vy, vz = vel_real[i]
                fx, fy, fz = force_real[i]
                pid = ids[i]

                f.write(f'Ar {x:.5f} {y:.5f} {z:.5f} {vx:.5f} {vy:.5f} {vz:.5f} {fx:.5f} {fy:.5f} {fz:.5f} {pid}\n')

        # 3. npz 포멧 작성
        data_to_save = {
            'step': step,
            'pos': system.pos,
            'vel': system.vel,
            'force': system.force,
            'box': system.box_size
        }

        file_name = self.filepath.parent / f'frame_{step:08d}.npy'
        np.save(file_name, data_to_save)