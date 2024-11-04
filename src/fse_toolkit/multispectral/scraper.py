from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..utils.common import find_nearest_station
from .utils import WeatherDataScraper, WebDriverSetup


def fetch_weather_data(
    user_lat: float,
    user_lon: float,
    items: List[str],
    start_time: str,
    end_time: str,
    temp_dir: str = "./temp"
) -> Dict[str, pd.DataFrame]:
    """
    獲取指定位置和時間範圍的天氣數據。

    Args:
        user_lat: 緯度
        user_lon: 經度
        items: 需要獲取的天氣數據項目
        start_time: 開始時間
        end_time: 結束時間
        temp_dir: 臨時文件目錄

    Returns:
        Dict containing weather data and station information
    """
    try:
        # 找到最近的站點
        nearest_stations = find_nearest_station(user_lat, user_lon)

        # 設置WebDriver
        temp_path = Path(temp_dir)
        if not temp_path.exists():
            temp_path.mkdir(parents=True)
        webdriver_setup = WebDriverSetup(Path(temp_dir))

        # 創建爬蟲實例
        scraper = WeatherDataScraper(webdriver_setup.browser, webdriver_setup.temp_dir_path)

        # 獲取天氣數據
        weather_data = None
        for i in range(len(nearest_stations)):
            try:
                weather_data = scraper.get_weather_data(
                    station={
                        'station_name': nearest_stations.loc[i, 'station_name'],
                        'area_name': nearest_stations.loc[i, 'area_name']
                    },
                    items=items,
                    start_time=start_time,
                    end_time=end_time
                )
                break
            except (ConnectionError, TimeoutError, ValueError, pd.errors.EmptyDataError) as e:
                print(f"An error occurred: {e}, finding next station...")

        return {
            "data": weather_data.set_index('觀測時間'),
            "station_info": nearest_stations.iloc[i]
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

    finally:
        if 'webdriver_setup' in locals():
            webdriver_setup.cleanup()
