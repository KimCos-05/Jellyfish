from pathlib import Path
import logging
import json
from dataclasses import asdict

import time
from datetime import timedelta

import cupy as cp

from script.utility.config import Config, getConfig
from script.utility.logger import set_logging_path, setup_logging

# 최상위 폴더의 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"

class FileManager:

    """파일 관리 manager"""

    def __init__(self, result_root: Path):
        self.result_root = result_root

        self.run_id = self._get_next_run_id()
        self.run_dir = self.result_root / f'run_{self.run_id}'

        self.run_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f'결과 폴더를 생성하였습니다 : {self.run_dir}')

        self.log_root = self.run_dir / 'logs'
        self.log_root.mkdir(parents=True, exist_ok=True)
        set_logging_path(self.log_root)
        logging.info(f'로그 파일 위치 : {self.log_root}')

        self.result_dir = self.get_trajectory_path()
        self.result_root = self.result_dir.parent

        self.result_root.mkdir(parents=True, exist_ok=True)
        logging.info(f'결과 파일 폴더를 생성하였습니다 : {self.result_root}')

    def _get_next_run_id(self) -> int:
        
        if not self.result_root.exists():
            return 1

        run_dirs = list(p for p in self.result_root.glob('run*') if p.is_dir())
  
        last_num = 0
        for d in run_dirs:
            try:
                num = int(d.name.split('_')[1])
                if num > last_num:
                    last_num = num

            except(IndexError, ValueError):
                continue

        return last_num + 1
    
    def save_config(self, config: Config):
        """config를 json으로 변환"""
        
        config_root = self.run_dir / 'config'
        config_path = config_root / 'config.json'

        config_root.mkdir(parents=True, exist_ok=True)

        try:
            config_dict = asdict(config)

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4, ensure_ascii=False)
            
            logging.info(f'config 파일을 {config_path}에 저장하였습니다.')

        except Exception as e:
            logging.error(f'오류 발생 : {e}')

    def get_trajectory_path(self, filename: str = "simulation.xyz") -> Path:
        """결과 저장용 폴더 경로 반환"""
        return self.run_dir / 'result' / filename

class SimulationMonitor:
    """시뮬레이션 진행 상황 관리 매니저"""

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.start_time = None
        self.last_time = None
        self.current_step = 0

    def start(self):
        """타이머 시작"""
        self.start_time = time.time()
        self.last_time = self.start_time
        logging.info('시뮬레이션 모니터링 시작')

    def print_status(self, step:int):
        """현재 진행 상황 출력"""
        if self.start_time is None:
            self.start()

        current_time = time.time()
        elapsed_total = current_time - self.start_time

        if elapsed_total == 0:
            return
        
        progress = (step / self.total_steps) * 100

        sps = step / elapsed_total

        remaining_steps = self.total_steps - step
        eta_seconds = remaining_steps / sps if sps > 0 else 0

        elapsed_str = str(timedelta(seconds=int(elapsed_total)))
        eta_str = str(timedelta(seconds = int(eta_seconds)))

        log_msg = (
            f"진행도: {progress:5.1f}% | "
            f"Step: {step}/{self.total_steps} | "
            f"속도: {sps:.1f} step/s | "
            f"경과: {elapsed_str} | "
            f"예상종료: {eta_str} 후"
        )

        logging.info(f'시뮬레이션 진행 상황 : {log_msg}')

class DeviceManager:
    """장치 관리"""

    @staticmethod
    def get_gpu_info():

        if not cp.is_available():
            return "GPU가 탐지되지 않음"
        
        try:
            dev_id = cp.cuda.Device().id
            props = cp.cuda.runtime.getDeviceProperties(dev_id)

            gpu_name = props['name'].decode('utf-8')

            free_mem, total_mem = cp.cuda.Device(dev_id).mem_info
            total_gb = total_mem / (1024**3)
            free_gb = free_mem / (1024**3)
            used_gb = total_gb - free_gb

            return f"GPU 정보 | {gpu_name} (VRAM: {used_gb:.3f}GB / {total_gb:.3f}GB | {free_gb:.3f}GB 남음)"
            
        except Exception as e:
            return f"(Error: {e})"

if __name__ == "__main__":
    setup_logging()
    filemanager = FileManager(RESULTS_ROOT)
    print(filemanager)

    sim_file = "simulation.json"
    config = getConfig("simulation.json")

    filemanager.save_config(config)
