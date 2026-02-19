from dataclasses import dataclass
from typing import Dict, List

import traceback
import logging

from pathlib import Path
import json

# 파일 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_ROOT = PROJECT_ROOT / "config"

# === define class of SimulationConfig ===
@dataclass
class MoleculeInfo:
    type: str
    count: int
    file: str

@dataclass
class SystemParams:
    box_size_nm: List[float]
    Temp_K: float
    boundary: str

@dataclass
class SimulationParams:
    dt_fs: float
    total_steps: int
    dlog: int
    GPU_block_size: int

@dataclass
class SimulationConfig:
    system: SystemParams
    simulation: SimulationParams
    molecules: List[MoleculeInfo]
    reactions: str

# === define class of MoleculeConfig ===
@dataclass
class MoleculeLJData:
    epsilon_K: float
    sigma_nm: float

@dataclass
class MoleculeConfig:
    name: str
    mass_amu: float
    LJ_params: MoleculeLJData

# === define class of ReactionConfig ===
@dataclass
class ReactionInfo:
    reactants: Dict[str, int]
    products: Dict[str, int]
    activate_energy_KJ_per_mol: float
    heat_of_reaction: float

@dataclass
class ReactionConfig:
    reactions: Dict[str, ReactionInfo]

# === 전체 config ===
@dataclass
class Config:
    simulation: SimulationConfig
    reactions: ReactionConfig
    molecules: Dict[str, MoleculeConfig]

# json 읽기 구문
def load_json_file(file_path: Path):

    if not file_path.exists():
        raise FileNotFoundError(f'파일이 없습니다: {file_path}')
    
    return json.loads(file_path.read_text(encoding='utf-8'))

# Config 읽기 함수

# 시뮬레이션 config 읽기
def load_simulation_config(file_name):

    # simulation config 가져오기
    target_path = CONFIG_ROOT / "simulation" / file_name
    simulation_config = load_json_file(target_path)

    system_params = SystemParams(**simulation_config['system'])
    simulation_params = SimulationParams(**simulation_config['simulation'])
    molecule_params = [MoleculeInfo(**m) for m in simulation_config['molecules']]

    return SimulationConfig(
        system = system_params,
        simulation = simulation_params,
        molecules = molecule_params,
        reactions = simulation_config['reactions']
    )

# reaction config 읽기
def load_reaction_config(file_name):

    # reaction config 가져오기
    target_path = CONFIG_ROOT / "reactions" / f'{file_name}.json'
    reaction_config = load_json_file(target_path)

    reaction_data = {}
    for rxn_name, rxn_data in reaction_config.items():
        reaction_data[rxn_name] = ReactionInfo(**rxn_data)

    return ReactionConfig(
        reactions = reaction_data
    )

# molecule config 읽기
def load_molecule_config(file_name):

    # molecule config 가져오기
    target_path = CONFIG_ROOT / "molecules" / f'{file_name}.json'
    molecule_config = load_json_file(target_path)

    LJ_params = MoleculeLJData(**molecule_config['LJ_params'])

    return MoleculeConfig(
        name = molecule_config['name'],
        mass_amu = molecule_config['mass_amu'],
        LJ_params = LJ_params
    )


# === config 읽기 함수 ===

# simulation.json 읽기 | system, simulation 파라미터 가져옴.
# -> molecule과 reaction의 파일경로 받아오기 -> molecule 읽기 -> reaction 읽고 molecule을 모두 읽어왔는지 검증하기.

def load_config(sim_file) -> Config:

    logging.info(f'프로젝트 루트: {PROJECT_ROOT}')

    try:
        # simulation config
        config_simulation = load_simulation_config(sim_file)

        logging.info('-- simulation config--')

        logging.info(f'Temp: {config_simulation.system.Temp_K}')
        logging.info(f'reaction file: {config_simulation.reactions}')
        for mol in config_simulation.molecules:
            logging.info(f'분자 종류: {mol.type}, 개수: {mol.count}')

        logging.info('')

        # reactions config  
        config_reactions = load_reaction_config(config_simulation.reactions)
        logging.info('-- reaction config--')
        for name, info in config_reactions.reactions.items():
            logging.info(f'Reaction name: {name}')
            logging.info(f'활성화 에너지: {info.activate_energy_KJ_per_mol}')

            for mol_name, count in info.reactants.items():
                logging.info(f'반응물: {mol_name}, {count}개')

            for mol_name, count in info.products.items():
                logging.info(f'생성물: {mol_name}, {count}개')

        logging.info('')

        # molecules config  
        logging.info('-- molecules config--')

        molecules = dict()
        for mol in config_simulation.molecules:
            molecule_file_name = mol.file
            molecule_config = load_molecule_config(molecule_file_name)

            molecules[mol.type] = molecule_config

        logging.info(molecules)
        logging.info(' ')

        return Config(
            simulation=config_simulation,
            reactions=config_reactions,
            molecules=molecules
        )
            
    except Exception:
        traceback.print_exc()
        logging.error(f'error')
        return None

# === reactions에 존재하는 모든 molecule 데이터를 가져왔는지 확인 ===

def check_config(config):
    logging.info('config 무결성 검사 시행')

    needed_molecules = set()

    for rxn in config.reactions.reactions.values():
        needed_molecules.update(rxn.reactants.keys())
        needed_molecules.update(rxn.products.keys())

    logging.info(f'need molecules: {needed_molecules}')

    defined_molecules = set(config.molecules.keys())

    missing_molecules = needed_molecules - defined_molecules

    if missing_molecules:
        logging.error(f'다음 분자의 설정 파일이 지정되어있지 않음 : {missing_molecules}')
        return False
    
    logging.info('검사 완료: 모든 분자가 정상적으로 정의됨')
    return True

# === config 불러오는 함수(불러오기 + 검사) ===

def getConfig(sim_file):
    config = load_config(sim_file)

    if config is None:
        logging.info('config 로딩 실패')
        exit(1)

    if not check_config(config):
        logging.info('설정 무결성 오류')
        exit(1)

    return config

# === 직접 실행 호출부 ===

if __name__ == "__main__":
    
    sim_file = "simulation.json"
    getConfig(sim_file)  