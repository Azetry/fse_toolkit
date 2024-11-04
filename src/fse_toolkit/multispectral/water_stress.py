from typing import Dict

import ee
import geemap


def mask_s2_clouds(image: ee.Image) -> ee.Image:
    ''' Mask clouds in Sentinel-2 image
    image: ee.Image, the Sentinel-2 image
    return: ee.Image, the Sentinel-2 image with clouds masked
    '''
    # 選取 QA60 band
    qa = image.select('QA60')

    # 定義 cloud 和 cirrus 的 bitmask
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    # 應用 bitmask 過濾雲層
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))

    # 更新影像掩碼並將影像除以 10000
    return image.updateMask(mask).divide(10000)


def calculate_lsw(image: ee.Image) -> ee.Image:
    ''' Calculate Land Surface Water Index (LSWI) from Sentinel-2 image
    image: ee.Image, the Sentinel-2 image
    return: ee.Image, the Sentinel-2 image with LSWI band
    '''
    # 選取 NIR (B8) 和 SWIR1 (B11) band
    nir = image.select('B8')
    swir1 = image.select('B11')

    # 計算 LSWI
    lswi = nir.subtract(swir1).divide(nir.add(swir1)).rename('LSWI')

    # 將 LSWI 作為新的一個 band 添加到影像中
    return image.addBands(lswi)


def calculate_wstress_from_ic(image_collection: ee.ImageCollection) -> ee.ImageCollection:
    ''' Calculate Water Stress Index (Wstress) from Sentinel-2 image collection
    image_collection: ee.ImageCollection, the Sentinel-2 image collection
    return: ee.ImageCollection, the Sentinel-2 image collection with Wstress band
    '''
    # 計算影像集合中的最大 LSWI (LSWImax)
    lswi_max = image_collection.select('LSWI').max()

    # 定義內部函數來計算每個影像的水壓指數 (Wstress)
    def calculate_wstress_per_image(image):
        lswi = image.select('LSWI')
        wstress = lswi.add(1).divide(lswi_max.add(1)).rename('Wstress')
        return image.addBands(wstress)

    # 應用 Wstress 計算到影像集合中的每個影像
    return image_collection.map(calculate_wstress_per_image)


def generate_wstress_composite(wstress_collection: ee.ImageCollection, aoi: ee.Geometry) -> ee.Image:
    ''' Generate a composite image of the Water Stress Index (Wstress) for the specified area of interest (AOI).
    This function calculates the median composite of the Wstress band from the provided image collection and clips it to the given AOI.
    wstress_collection: ee.ImageCollection, the Sentinel-2 image collection with Wstress band
    aoi: ee.Geometry, the area of interest
    return: ee.Image, the composite image of Wstress clipped to the AOI
    '''
    # 計算中位數合成並剪裁到感興趣區域 (aoi)
    wstress_composite = wstress_collection.select('Wstress').median().clip(aoi)
    return wstress_composite


def calculate_water_stress(bounds: Dict[str, float], start_date: str, end_date: str, output_numpy=True):
    # Load and preprocess Sentinel-2 imagery for August of three years
    # Only August
    aoi = ee.Geometry.BBox(bounds['left'], bounds['bottom'], bounds['right'], bounds['top'])
    s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
        .filterBounds(aoi)\
        .filterDate(start_date, end_date)\
        .filter(ee.Filter.calendarRange(8, 8, 'month'))\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))


    # calendarRange: 僅選取八月
    s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
        .filterBounds(aoi)\
        .filterDate(start_date, end_date)\
        .filter(ee.Filter.calendarRange(8, 8, 'month'))\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
        .map(mask_s2_clouds)\
        .map(calculate_lsw)

    wstress_collection = calculate_wstress_from_ic(s2_collection)
    wstress_composite = generate_wstress_composite(wstress_collection, aoi)

    if output_numpy:
        # Convert the Wstress composite to a numpy array
        wstress_array = geemap.ee_to_numpy(
                    wstress_composite,
                    region=aoi,
                    bands=['Wstress'],
                    scale=20
                )
        return wstress_array
    else:
        return wstress_composite
