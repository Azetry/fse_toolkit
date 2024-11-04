'''
This module provides functions to calculate NDVI, SR, and FPAR from NIR and Red bands.
'''
import numpy as np


def calculate_ndvi(nir_band, red_band, lower_bound=0.275, upper_bound=0.97):
    '''
    Calculate NDVI from NIR and Red bands.
    - devided by zero will be NaN
    - It's fine since it's not actually a tree. (NIR=0 and RED=0)
    - If NDVI < lower_bound, it is considered as no vegetation. (0.275)
    - If NDVI > upper_bound, set to upper_bound (if NDVI == 1, SR will be infinite)
    Args:
        nir_band: numpy array, NIR band.
        red_band: numpy array, Red band.
    Return:
        np.array, NDVI image
    '''
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir_band - red_band) / (nir_band + red_band)

    # ndvi_filtered = np.where(ndvi >= lower_bound, ndvi, np.nan)
    # ndvi_filtered = np.where(ndvi_filtered == 1, upper_bound, ndvi_filtered)

    # 如果 ndvi 小於 lower_bound，則將其設為 NaN, 如果 ndvi 大於等於 upper_bound，則將其設為 upper_bound
    ndvi_filtered = np.where(ndvi > lower_bound, ndvi, np.nan)
    ndvi_filtered = np.where(ndvi_filtered >= upper_bound, upper_bound, ndvi_filtered)


    return ndvi_filtered


def calculate_sr(ndvi):
    '''
    Calculate SR from NDVI.
    Args:
        ndvi: np.array, NDVI image
    Return:
        np.array, SR image
    '''
    with np.errstate(divide='ignore', invalid='ignore'):
        sr = (1 + ndvi) / (1 - ndvi)
    return sr


def calculate_fpar(ndvi, sr, fpar_type='loss', fpar_min=0.001, fpar_max=0.95):
    '''
    Calculate FPAR from NDVI and SR.
    Args:
        ndvi: np.array, NDVI image
        sr: np.array, SR image
        fpar_type: str, 'field', 'potter', or 'loss'
        fpar_min: float, minimum FPAR value
        fpar_max: float, maximum FPAR value
    Return:
        np.array, FPAR image (loss)
    '''
    ndvi_min = np.nanmin(ndvi)
    ndvi_max = np.nanmax(ndvi)
    sr_min = np.nanmin(sr)
    sr_max = np.nanmax(sr)

    fpar_field = ((ndvi - ndvi_min) / (ndvi_max - ndvi_min)) * (fpar_max - fpar_min) + fpar_min

    fpar_potter = ((sr - sr_min) / (sr_max - sr_min)) * (fpar_max - fpar_min) + fpar_min

    fpar_loss = (fpar_field + fpar_potter) / 2

    match fpar_type:
        case 'field':
            return fpar_field
        case 'potter':
            return fpar_potter
        case 'loss':
            return fpar_loss
        case _:
            raise ValueError('Invalid type. Choose from "field", "potter", or "loss".')
