from typing import Union

import numpy as np


# epsilon is np.array or float
def calculate_npp(apar: np.array, epsilon: Union[np.array, float]) -> np.array:
    # NPP = APAR * epsilon
    return apar * epsilon

def calculate_apar(fpar_loss: np.array, sol: Union[np.array, float], scaler=0.5)-> np.array:
    return fpar_loss * sol * scaler

def caculate_epsilon(water_stress, te1, te2, epsilon_max=0.985):
    """
    Calculate the epsilon value based on water stress and transpiration efficiency.
    Args:
        water_stress (float): The average water stress value.
        te1 (float): The first transpiration efficiency value.
        te2 (float): The second transpiration efficiency value.
        epsilon_max (float, optional): The maximum epsilon value. Defaults to 0.985.
    Returns:
        float: The calculated epsilon value.
    """
    # 原本 water_stress 是要矩陣乘上其他數值，但是為了方便計算，water_stress 我們取平均值輸入，因此 epsilon 也是一個數值
    # 我們假設樹種為闊葉林，epsilon_max 設為 0.985

    return water_stress*te1*te2*epsilon_max
