'''
te1 and te2 are the functions that calculate the temperature effect on the forest carbon evaluation.
'''
import numpy as np


def calculate_te1(t_opt):
    """
    Calculate the Te1 value based on T_opt (the optimal temperature in June).
    :param t_opt: The optimal temperature in June for location x
    :return: Calculated Te1 value
    """
    return 0.8 + 0.02 * t_opt - 0.0005 * (t_opt ** 2)

def calculate_term1(t_opt, t):
    ''' Compute the first term of Te2 calculation.
    Args:
        t_opt (float): The optimal temperature.
        t (float): The current temperature.
    Returns:
        float: The computed value of the first term.
    '''
    exponent1 = 0.2 * (t_opt - 10 - t)
    return 1 + np.exp(exponent1)

def calculate_term2(t_opt, t):
    ''' Compute the second term of Te2 calculation.
    Args:
        t_opt (float): The optimal temperature.
        T (float): The current temperature.
    Returns:
        float: The computed value of the second term.
    '''
    exponent2 = 0.3 * (-t_opt - 10 + t)
    return 1 + np.exp(exponent2)

def calculate_te2(t_opt, t):
    ''' Calculate Te2 using both terms.
    Args:
        t_opt (float): The optimal temperature.
        T (float): The current temperature.
    Returns:
        float: Calculated Te2 value.
    '''
    term1 = calculate_term1(t_opt, t)
    term2 = calculate_term2(t_opt, t)
    return 1.1814 / (term1 * term2)
