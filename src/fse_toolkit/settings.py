from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]

# Constants for Weather data fetching
STATION_RECORD_PATH = PROJECT_ROOT/"data/station_record.csv"
STATION_URL = 'https://agr.cwa.gov.tw/NAGR/history/station_month'

# Constants for Water stress calculation using Google Earth Engine
EE_ACCOUNT = 'my-earth@fses-ee.iam.gserviceaccount.com'
EE_PRIVATE_KEY_FILE = PROJECT_ROOT / 'data/fses-ee-8b310a8304b0.json'
