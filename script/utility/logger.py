from pathlib import Path
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

def setup_logging():
    logging.basicConfig(level = logging.INFO, 
                        format = LOG_FORMAT, 
                        handlers = [logging.StreamHandler()])

def set_logging_path(run_dir: Path):
    log_file = run_dir / "simulation.log"

    file_handler = logging.FileHandler(log_file, encoding= 'utf-8')
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    file_handler.setLevel(logging.INFO)

    logging.getLogger().addHandler(file_handler)

    logging.info(f'로깅 설정 완료. 로그 파일 저장 위치 : {log_file}')