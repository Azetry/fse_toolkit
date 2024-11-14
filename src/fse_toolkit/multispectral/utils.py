"""
Utils, Forest Carbon Sink Evaluation, Multispectral
- WebDriver setup
- Weather data scraping
"""
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import Chrome, ChromeOptions, ChromeService
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager

from ..settings import STATION_URL


# used for water stress
def set_date_range(year_now, month=8, years=3):
    ''' Set the date range for the given year and month
    year_now: int, the current year
    month: int, the month
    years: int, the duration of the date range (years)
    return: tuple, the start date and end date
    '''
    next_month = datetime(year_now, month, 28) + timedelta(days=4)
    end_date = next_month - timedelta(days=next_month.day)
    start_date = datetime(year_now-years+1, month, 1)

    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


# used for weather data scraping
class WebDriverSetup:
    '''
    Sets up the Chrome WebDriver with the specified options.
    '''
    def __init__(self, temp_root: Path):
        self.temp_root = temp_root
        self.temp_dir = self.create_temp_dir()
        self.temp_dir_path = str(Path(self.temp_dir.name).resolve())
        self.driver_path = self.install_chrome_driver()
        self.browser = self.setup_webdriver()
        print(f"WebDriver setup completed. Temp dir: {self.temp_dir_path}")

    def create_temp_dir(self):
        """Creates a temporary directory for downloads."""
        return tempfile.TemporaryDirectory(dir=self.temp_root)

    def install_chrome_driver(self) -> str:
        """Installs ChromeDriver using webdriver_manager."""
        return ChromeDriverManager().install()

    def setup_webdriver(self) -> Chrome:
        """Sets up the Chrome WebDriver with the specified options."""
        options = ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')  # 停用GPU加速
        options.add_argument('--disable-software-rasterizer')  # 停用軟體光柵化
        options.add_argument('--remote-debugging-port=9222')  # 新增除錯埠
        options.add_experimental_option("prefs", {
            "download.default_directory": self.temp_dir_path,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        })
        service = ChromeService(executable_path=self.driver_path)
        return Chrome(service=service, options=options)

    def cleanup(self):
        """Quits the browser and cleans up the temporary directory."""
        if self.browser:
            self.browser.quit()
        if self.temp_dir:
            self.temp_dir.cleanup()

# used for weather data scraping
class WeatherDataScraper:
    '''
    A class to scrape weather data from the Taiwan Agricultural Research Institute website
    '''
    def __init__(self, driver: WebDriver, temp_dir_path: str):
        self.driver = driver
        self.temp_dir_path = temp_dir_path
        self.base_url = STATION_URL

    def validate_inputs(self, station: Dict[str, str],
                        items: List[str],
                        start_time: str,
                        end_time: str
                        ):
        ''' Validate the inputs for the scraper
        station: dict, the station information
        items: list, the items to scrape
        start_time: str, the start time in the format of 'yyyy-mm'
        end_time: str, the end time in the format of 'yyyy-mm'
        raise: ValueError, if the inputs are invalid
        '''
        if 'station_name' not in station or 'area_name' not in station:
            raise ValueError("station must have 'station_name' and 'area_name' fields")
        if not self.is_valid_date_format(start_time, '%Y-%m') or \
            not self.is_valid_date_format(end_time, '%Y-%m'):
            raise ValueError("start_time and end_time must be in the format \
                            of 'yyyy-mm'")
        if not items:
            raise ValueError("items list cannot be empty")

    @staticmethod
    def is_valid_date_format(date_str: str, date_format: str) -> bool:
        ''' Check if the date string is in the correct format
        date_str: str, the date string
        date_format: str, the date format
        return: bool, True if the date string is in the correct format
        '''
        try:
            time.strptime(date_str, date_format)
            return True
        except ValueError:
            return False

    def navigate_to_site(self):
        ''' Navigate to the website
        raise: ValueError, if the website is not accessible
        '''
        try:
            self.driver.get(self.base_url)
        except Exception as exc:
            raise ValueError(f"Cannot access the website: {self.base_url}") from exc

    def select_station(self, station: Dict[str, str]):
        ''' Select the station from the dropdown
        station: dict, the station information
        '''
        self.select_from_dropdown('station_level', '農業站')
        self.select_from_dropdown('area_name', station['area_name'])
        self.select_from_dropdown('station_name', station['station_name'])

    def select_items(self, items: List[str]):
        ''' Select the items from the dropdown
        items: list, the items to select
        raise: ValueError, if none of the specified items were found in the list
        '''
        item_multiple = Select(self.driver.find_element(By.ID, 'item_multiple'))
        for item in items:
            try:
                item_multiple.select_by_visible_text(item)
                time.sleep(0.5)
            except NoSuchElementException as e:
                print(e)
        if len(item_multiple.all_selected_options) == 0:
            raise ValueError("None of the specified items were found in the list")

    def set_date_range(self, start_time: str, end_time: str):
        ''' Set the date range for the report
        start_time: str, the start time in the format of 'yyyy-mm'
        end_time: str, the end time in the format of 'yyyy-mm'
        '''
        self.set_date_field("start_time", start_time)
        self.set_date_field("end_time", end_time)

    def set_date_field(self, field_id: str, date_value: str):
        ''' Set the date field value
        field_id: str, the id of the date field
        date_value: str, the date value
        '''
        date_field = self.driver.find_element(By.ID, field_id)
        date_field.clear()
        self.driver.execute_script(f"arguments[0].value = '{date_value}';", date_field)
        date_field.send_keys(Keys.ESCAPE)
        date_field.clear()
        self.driver.execute_script(f"arguments[0].value = '{date_value}';", date_field)

    def download_report(self):
        ''' Download the report '''
        csv_time_radio = self.driver.find_element(By.ID, "radios-3")
        csv_time_radio.click()
        download_btn = self.driver.find_element(By.ID, "create_report")
        download_btn.click()

    def read_downloaded_file(self) -> pd.DataFrame:
        ''' Read the downloaded file and return it as a DataFrame
        return: pd.DataFrame, the report data
        '''
        time.sleep(1)
        downloaded_files = os.listdir(self.temp_dir_path)
        if not downloaded_files:
            raise FileNotFoundError("No files found in the download directory")
        report_df = pd.read_csv(
            os.path.join(self.temp_dir_path, downloaded_files[0]),
            encoding='big5',
            skiprows=1
        )
        return report_df

    def select_from_dropdown(self, dropdown_name: str, visible_text: str):
        ''' Select an option from the dropdown
        dropdown_name: str, the name of the dropdown
        visible_text: str, the visible text of the option
        raise: ValueError, if the option is not found in the dropdown
        '''
        dropdown = Select(self.driver.find_element(By.NAME, dropdown_name))
        try:
            dropdown.select_by_visible_text(visible_text)
            time.sleep(0.5)
        except Exception as exc:
            raise ValueError(f"Cannot find the option '{visible_text}' \
                            in dropdown '{dropdown_name}'") from exc

    def get_weather_data(self, station: Dict[str, str],
                        items: List[str],
                        start_time: str,
                        end_time: str) -> pd.DataFrame:
        ''' Get the weather data from the website
        station: dict, the station information
        items: list, the items to scrape
        start_time: str, the start time in the format of 'yyyy-mm'
        end_time: str, the end time in the format of 'yyyy-mm'
        return: pd.DataFrame, the weather data
        '''
        self.validate_inputs(station, items, start_time, end_time)
        self.navigate_to_site()
        self.select_station(station)
        self.select_items(items)
        self.set_date_range(start_time, end_time)
        self.download_report()
        return self.read_downloaded_file()
