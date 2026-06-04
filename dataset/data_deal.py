import pandas as pd
import numpy as np

# Details of the dataset
# Volume min(0) max(7280)
# Unemployment min(54) max(612)   1972-2025年
# AirQualityUCI  C6H6(GT) min(0.1490477388337664)  max(63.74147644829163)
# ILINet min(27263) max(2260794)   1997-2023年
# Foursquare  users 122042/data stream length: 432
# Syn1 min(1) max(1000)
# TDrive_new : new TDrive dataset

# Foursquare dataset deal process
def data_deal_Four(df,df2):
     all_countries = df2['Country code'].unique()
     new = df[df['User ID'] == 2]
     all_dates_str = sorted(list(dict.fromkeys(new['date'].values.tolist())))
     print(all_dates_str, len(all_dates_str))

     # new DataFrame
     filled_df_list = []
     UIDs = df['User ID'].unique()
     for uid in UIDs:
          df_user = df[df['User ID'] == uid].copy()
          user_date_country = dict(zip(df_user['date'], df_user['Country code']))

          filled_records = []
          for d in all_dates_str:
               if d in user_date_country:
                    country = user_date_country[d]
               else:
                    country = np.random.choice(all_countries)
               filled_records.append({
                    'User ID': uid,
                    'date': d,
                    'Country code': country
               })
          filled_df_list.append(pd.DataFrame(filled_records))

     # combined result
     df_filled = pd.concat(filled_df_list, ignore_index=True)
     df_filled = df_filled.sort_values(['User ID', 'date']).reset_index(drop=True)
     # df_filled.to_csv('data_new.csv', index=False)

     return df_filled


# TDrive dataset deal
# Divide the entire map into sections
def TDrive_deal(df):

     min_lon, max_lon = df["Longitude"].min(), df["Longitude"].max()
     min_lat, max_lat = df["Latitude"].min(), df["Latitude"].max()
     num_lon_bins = 5
     num_lat_bins = 2
     lon_bins = np.linspace(min_lon, max_lon, num_lon_bins + 1)
     lat_bins = np.linspace(min_lat, max_lat, num_lat_bins + 1)
     grid_list = []

     for lat_i in range(num_lat_bins):
          for lon_i in range(num_lon_bins):
               LID = lat_i * num_lon_bins + lon_i + 1

               lon_min = lon_bins[lon_i]
               lon_max = lon_bins[lon_i + 1]

               lat_min = lat_bins[lat_i]
               lat_max = lat_bins[lat_i + 1]

               grid_list.append({
                    "LID": LID,
                    "lon_min": lon_min,
                    "lon_max": lon_max,
                    "lat_min": lat_min,
                    "lat_max": lat_max
               })
     grid_df = pd.DataFrame(grid_list)
     grid_df.to_csv("Grid_Definition.csv", index=False)
     return df

# The entire area is divided into multiple grids
def grid(df):
     min_lon, max_lon = df["Longitude"].min(), df["Longitude"].max()
     min_lat, max_lat = df["Latitude"].min(), df["Latitude"].max()
     num_lon_bins = 5
     num_lat_bins = 2
     lon_bins = np.linspace(min_lon, max_lon, num_lon_bins + 1)
     lat_bins = np.linspace(min_lat, max_lat, num_lat_bins + 1)
     grid_list = []
     for lat_i in range(num_lat_bins):
          for lon_i in range(num_lon_bins):
               LID = lat_i * num_lon_bins + lon_i + 1

               lon_min = lon_bins[lon_i]
               lon_max = lon_bins[lon_i + 1]

               lat_min = lat_bins[lat_i]
               lat_max = lat_bins[lat_i + 1]

               grid_list.append({
                    "LID": LID,
                    "lon_min": lon_min,
                    "lon_max": lon_max,
                    "lat_min": lat_min,
                    "lat_max": lat_max
               })
     grid_df = pd.DataFrame(grid_list)
     grid_df.to_csv("TDrive_Grid.csv", index=False)

     return grid_df



