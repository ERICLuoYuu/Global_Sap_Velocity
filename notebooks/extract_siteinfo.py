# Add parent directory to Python path
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from src.tools import extract_site_info

def main():
  data_dir = Path('./data/raw/0.1.5/0.1.5/csv/sapwood')

  site_info = extract_site_info(data_dir)
  print(site_info.head())
  print(f"Site information extracted: {site_info}")
  site_info.to_csv('./data/raw/0.1.5/0.1.5/csv/sapwood/site_info.csv', index=False)
  site_info = pd.read_csv('./data/raw/0.1.5/0.1.5/csv/sapwood/site_info.csv')
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.scatter(site_info['lon'], site_info['lat'])
  ax.set_xlabel('Longitude')
  ax.set_ylabel('Latitude')
  ax.set_title('Site Locations')
  plt.show()
if __name__ == "__main__":
    main()