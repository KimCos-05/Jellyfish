import sys
import logging
import numpy as np
from pathlib import Path

# 최상위 폴더의 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.append(str(PROJECT_ROOT))

from script.utility.config import getConfig
from script.utility.logger import setup_logging
from script.utility.UnitConverter import UnitConverter
from script.utility.manager import FileManager, SimulationMonitor
from script.core.simulation_generater import SystemState
from script.core.integrator import Integrator
from script.analysis.archiving import Archiving

# === 시뮬레이션 시작 함수 ===

def start_simulation(config_name='simulation.json'):

    # 1. 시뮬레이션 로그 및 파일 경로 설정
    setup_logging()
    file_manager = FileManager(PROJECT_ROOT / "results")

    # 2. 시뮬레이션 config 읽기 및 복제
    config = getConfig(config_name)
    file_manager.save_config(config)

    # 3. 시뮬레이션 단위 변환 / 현재는 첫 분자를 기준으로 함.
    first_mol_key = list(config.molecules.keys())[0]
    ref_mol = config.molecules[first_mol_key]

    converter = UnitConverter(
        mass_ref = ref_mol.mass_amu * 1.66054e-27,
        sigma_ref = ref_mol.LJ_params.sigma_nm * 1e-09,
        epsilon_ref = ref_mol.LJ_params.epsilon_K * 1.380649e-23
    )

    logging.info(converter.get_info_str())

    # 4. 입자 초기 설정
    system = SystemState.initialize_system(config, converter)

    logging.info('System State 설정 완료')
    logging.info(f'총 입자 수: {system.n_particles}, 계 크기 : {system.box_size}, 입자 0번 위치: {system.pos[0]}, 입자 0번 속도: {system.vel[0]}')

    # 5. CUDA kernel 불러오기
    integrator = Integrator(system, config, converter)

    # 6. 시뮬레이션 정보 설정
    total_steps = config.simulation.simulation.total_steps
    dlog = config.simulation.simulation.dlog
    target_temp = config.simulation.system.Temp_K

    traj_path = file_manager.get_trajectory_path('simulation_result.xyz')
    writer = Archiving(traj_path)

    logging.info(f'integrate 호출')
    logging.info(f'시뮬레이션 시작: 총 {total_steps} 스텝')

    # 7. 시뮬레이션 매니저 설정
    monitor =  SimulationMonitor(total_steps)
    monitor.start()

    # 8. 시뮬레이션 초기값 저장

    writer.write_frame(step=0, system=system, converter=converter, dt_fs=config.simulation.simulation.dt_fs)
    logging.info(f'0 Step 파일 저장 : {traj_path}')
    monitor.print_status(0)

    # 8. 시뮬레이션 메인 loop
    for s in range(0, total_steps, dlog):

        integrator.integrate(dlog, converter, thermostat_on=True)

        nows = s + dlog
        integrator.update_cpu_state(system)

        writer.write_frame(step=nows, system=system, converter=converter, dt_fs=config.simulation.simulation.dt_fs)
        logging.info(f'{nows} Step 파일 저장 : {traj_path}')
        monitor.print_status(nows)

    logging.info("시뮬레이션 종료")
    logging.info(f'완료 상태: Step {total_steps}')
    logging.info(f'입자 0번 위치: {system.pos[0]}')
    logging.info(f'입자 0번 속도: {system.vel[0]}')
    
if __name__ == "__main__":

    config_name = ['phase120K', 'phase130K', 'phase140K', 'phase150K', 'phase160K']

    for n in config_name:
        try:
            start_simulation(f'{n}.json')
        except Exception as e:
            logging.error(f'{n} 시뮬레이션 중 오류 발생 : {e}')
            continue
