import numpy as np
from dataclasses import dataclass

@dataclass
class UnitConverter:
    """
    실제 단위의 정규화 과정

    본 시뮬레이션의 핵심 계산은 총 2가지로,

    1. LJ 퍼탠셜의 계산(힘 구하기)
    2. 뉴턴 방정식의 사용

    을 사용함. 따라서 LJ퍼탠셜의 3가지 주요 인자인 (m, sigma, epsilon)을 각각 1로 변환하는 정규화를 통해 계산을 더욱 간단하게 만들 수 있음.

    따라서, 모든 물리량을 기준 입자(Reference Particle)의 질량(m), 거리(sigma), 에너지(epsilon)에 대한 비율로 표현함.
    """

    mass_ref: float    # [kg] 기준 질량
    sigma_ref: float   # [m] 기준 거리
    epsilon_ref: float # [J] 기준 에너지

    kB = 1.380649e-23  # [J/K] 볼츠만 상수

    def __post_init__(self):
        """
        초기화 이후 자동으로 실행되는 함수.
        핵심 3요소(m, sigma, epsilon)을 바탕으로 시간 단위(tau)를 유도.
        공식: tau = sigma * sqrt(m/epsion)
        """
 
        self.tau = self.sigma_ref * np.sqrt(self.mass_ref / self.epsilon_ref)

    def T_to_reduced(self, T_k: float) -> float:
        """[K] -> T* (T* = kB * T / epsilon)"""
        return (self.kB * T_k) / self.epsilon_ref
    
    def dt_to_reduced(self, dt_fs: float) -> float:
        """[fs] -> t* (t* = t / tau)"""
        dt_sec = dt_fs * 1.0e-15
        return dt_sec / self.tau
    
    def len_to_reduced(self, l_nm: float) -> float:
        """[nm] -> l* (l* = l / sigma)"""
        l_meter = l_nm * 1.0e-9
        return l_meter / self.sigma_ref
    
    def mass_to_reduced(self, m_amu: float) -> float:
        """[amu] -> m* (m* = m / m_ref)"""
        #1amu = 1.66054e-27kg
        m_kg = m_amu * 1.66054e-27
        return m_kg / self.mass_ref
    
    def get_info_str(self) -> str:
        """변환 정보를 문자열로 반환"""

        return(f'[단위 변환 정보]\n'
               f' 시간: {self.tau:.3e} s'
               f' 온도: {self.epsilon_ref / self.kB:.2} K'
               f' 길이: {self.sigma_ref * 1e9:.2f} nm')
    
    def reduced_to_real_time(self, t_star: float) -> float:
        """t* -> [s]"""
        t_sec = t_star * self.tau
        return t_sec * 1.0e15
    
    def reduced_to_real_temp(self, T_star: float) -> float:
        """T* -> [K]"""
        return (T_star * self.epsilon_ref) / self.kB
    
    def reduced_to_real_len(self, l_star: float) -> float:
        """m* -> [amu]"""
        l_meter = l_star * self.sigma_ref
        return l_meter * 1.0e9
    
    def reduced_to_real_energy(self, e_star: float) -> float:
        """E* -> [J]"""
        return e_star * self.epsilon_ref
    
    def reduced_to_real_vel(self, v_star: float) -> float:
        """v* -> [m/s] (v=v* * (sigma/tau))"""
        return v_star * (self.sigma_ref / self.tau)
    
    def reduced_to_real_force(self, f_star: float) -> float:
        """f* -> [N] (F= F* * (epsilon / sigma))"""
        return f_star * (self.epsilon_ref / self.sigma_ref)