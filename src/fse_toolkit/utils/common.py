"""
Common utils, Forest Carbon Sink Evaluation
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from ..settings import STATION_RECORD_PATH


def to_image(rgb_bands, display=False):
    ''' Convert 3 bands to rgb image
    rgb_bands: numpy array, 3 bands of the image (r, g, b)
    display: bool, show the image or not
    return: np.array, rgb image
    '''
    # Stack the bands
    rgb = np.stack(rgb_bands, axis=0)
    rgb = np.moveaxis(rgb, 0, -1)

    # Normalize the bands
    rgb = (rgb - np.nanmin(rgb)) / (np.nanmax(rgb) - np.nanmin(rgb))

    # Show the image(optional)
    if display:
        plt.imshow(rgb)
        plt.show()

    return rgb

def convert_grayscale_to_heatmap(array: np.array, output_png, item='ndvi'):
    """
    Convert grayscale array to heatmap
    Parameters:
    array: np.array, grayscale array
    output_png: str, output path
    item: str, item name
    """

    # 計算基於矩陣大小的適當圖片尺寸
    height, width = array.shape
    fig_width = 20  # 可以調整這個基準值
    fig_height = fig_width * (height / width)
    
    # 創建圖形
    plt.figure(figsize=(12, 8))

    # 繪製熱力圖
    img = plt.imshow(array,
                    cmap='RdYlGn',  # 使用紅黃綠色階
                    vmin=np.nanmin(array),        # NDVI的最小值
                    vmax=np.nanmax(array))         # NDVI的最大值

    # 添加色標
    plt.colorbar(img, label=f'{item.upper()} Value')

    # 設定標題
    # plt.title('NDVI Heatmap')

    # 移除座標軸
    plt.axis('off')

    # 調整布局
    plt.tight_layout()

    # 儲存圖片
    plt.savefig(output_png,
                dpi=300,
                bbox_inches='tight',
                pad_inches=0)

    # 關閉圖形
    plt.close()

    print(f"熱力圖已儲存至: {output_png}")


def get_lat_lon_bounds(src: rasterio.DatasetReader):
    ''' Get the lat, lon bounds of the image
    src: rasterio dataset
    return: tuple, (lat_top, lon_left, lat_bottom, lon_right)
    '''
    bounds = src.bounds
    crs = src.crs

    # EPSG:4326 is the lat, lon coordinate system
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    lon_left, lat_top = transformer.transform(bounds.left, bounds.top)
    lon_right, lat_bottom = transformer.transform(bounds.right, bounds.bottom)

    return lat_top, lon_left, lat_bottom, lon_right


def read_tif(tif_file):
    ''' Read a tif file
    tif_file: str, the path of the tif file
    return: dict, the information of the tif file
    '''
    # 使用 rasterio 打開 TIF 檔案
    with rasterio.open(tif_file) as src:
        # 建立一個 numpy 陣列來存放影像資料
        bands = np.zeros((src.count, src.height, src.width), dtype='float32')
        for i in range(1, src.count + 1):
            bands[i-1] = src.read(i)
        # red_band = src.read(1).astype('float32')  # 1st band
        # nir_band = src.read(4).astype('float32')  # 4th band

        # 獲取影像的地理變換參數
        transform = src.transform
        print("影像的地理變換矩陣:", transform)

        # 獲取 CRS (Coordinate Reference System)
        crs = src.crs
        print("影像的座標參考系統:", crs)

        # 獲取影像的範圍
        bounds = src.bounds
        lat_top, lon_left, lat_bottom, lon_right = get_lat_lon_bounds(src)
        center = (lat_top + lat_bottom) / 2, (lon_left + lon_right) / 2
        # 四個角與中心之經緯度座標
        coordinates = {
            "top": lat_top,
            "left": lon_left,
            "bottom": lat_bottom,
            "right": lon_right,
            "center": center
        }

        # Copy metadata for writing output files
        meta = src.meta.copy()

    return {
        "bands": bands,
        "transform": transform,
        "crs": crs,
        "bounds": bounds,
        "coordinates": coordinates,
        "meta": meta
    }


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on the earth."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c  # Radius of Earth in kilometers
    return km

def find_nearest_station(user_lat: float, user_lon: float, rank=5) -> pd.Series:
    """Find the nearest weather station based on user's latitude and longitude."""
    try:
        print(STATION_RECORD_PATH)
        station_record_df = pd.read_csv(STATION_RECORD_PATH)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'Station record file not found at {STATION_RECORD_PATH}') from exc

    if 'latitude' not in station_record_df.columns or 'longitude' not in station_record_df.columns:
        raise ValueError("Station record file does not contain 'latitude' and 'longitude' columns")

    distances = station_record_df.apply(
        lambda row: haversine(user_lat, user_lon, row['latitude'], row['longitude']),
        axis=1
    )
    # 回傳距離最近的前 rank 個站點
    nearest_station_idxs = distances.nsmallest(rank).index
    return station_record_df.iloc[nearest_station_idxs].reset_index()

def statistics(array: np.array):
    '''
    Calculate statistics of the given array.
    '''
    return {
        'max': np.nanmax(array),
        'min': np.nanmin(array),
        'mean': np.nanmean(array),
        'std': np.nanstd(array),
        'median': np.nanmedian(array),
        'count': np.count_nonzero(~np.isnan(array)),
        'sum': np.nansum(array)
    }
