import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm

# Syn2, Syn3, Foursquare, and TDrive datasets actual statistical results

def real_statistical_result(df, timestamp, domain):

    real_result = []
    for time in tqdm(timestamp):
        count_vector_real = np.zeros(len(domain), dtype=int)
        df_t = df[df['Time'] == time]
        values_t = df_t['LID'].values.tolist()
        category_count_real = Counter(values_t)
        cat2idx = {cat: i for i, cat in enumerate(domain)}
        for cat, cnt in category_count_real.items():
            idx = cat2idx[cat]
            count_vector_real[idx] = cnt
        real_result.append(count_vector_real)
    return real_result

def coumpute_result(df):

    timestamp = df['Time'].unique().tolist()
    domain = sorted(df['LID'].unique().tolist())
    print(domain, len(domain))
    real_result = real_statistical_result(df, timestamp, domain)
    df_real = pd.DataFrame()
    df_real['Time'] = timestamp
    df_real['real frequency'] = real_result

    return df_real


# if __name__ == "__main__":
#     Path = 'E:\Code\Stream data release LDP\dataset\\'
#     df = pd.read_csv(Path + 'TDrive_new.csv', encoding='unicode_escape')
#     df_real = coumpute_result(df)
#     df_real.to_json('Tdrive.json', orient='records')