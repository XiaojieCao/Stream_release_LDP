import numpy as np
import pandas as pd
import random
import oue,krr
from collections import Counter
from tqdm import tqdm


def Piecewise_mechanism(t, epsilon):

    C = (np.exp(epsilon/2)+1) / (np.exp(epsilon/2)-1)
    l = (C+1)/2 * t - (C-1)/2
    r = l + C -1
    x = random.uniform(0, 1)
    if x < np.exp(epsilon/2) / (np.exp(epsilon/2)+1):
        perturbed_t = random.uniform(l, r)
    else:
        x_1 = random.uniform(0, 1)
        if x_1 <= t:
            perturbed_t = random.uniform(- C, l )
        else:
            perturbed_t = random.uniform(r, C)
    perturbed_t = perturbed_t/C

    return perturbed_t


def perturb_krr(domain, input_value, epsilon):

    KRR = krr.KRR(domain, epsilon)
    perturbed_value = KRR.encode(tuple(input_value))

    return perturbed_value

def perturb_oue(domain, input_value, epsilon):

    OUE = oue.OUE(len(domain), epsilon)
    index = domain.index(input_value)
    perturbed_value = OUE.perturb(index)

    return perturbed_value

def single_user_perturbation(data, epsilon, w):

    noise_result = []
    for i in range(0, len(data), w):
        block_data = data[i:i + w]
        for value in block_data:
            noise_data = Piecewise_mechanism(value, epsilon/w)
            noise_result.append(noise_data)

    return noise_result

# dataset: data processing
def data_deal(df,column):

    max_value = df[column].max()
    min_value = df[column].min()
    # max_value, min_value = 1, 1000
    df_new = df.copy()
    df_new[column] = df_new[column].apply(lambda x: 2* (x - min_value) / (max_value - min_value) -1)
    input_data = df_new[column].values.tolist()

    return max_value, min_value, input_data


# multi-user
def real_statistical_result(df, timestamp, domain):

    real_result = []
    for time in timestamp:
        count_vector_real = np.zeros(len(domain), dtype=int)
        df_t = df[df['Time'] == time]
        values_t = df_t['Value'].values.tolist()
        category_count_real = Counter(values_t)
        cat2idx = {cat: i for i, cat in enumerate(domain)}
        for cat, cnt in category_count_real.items():
            idx = cat2idx[cat]
            count_vector_real[idx] = cnt
        real_result.append(count_vector_real)

    return real_result

def perturb_value(values_t,domain, epsilon_t):

    if len(domain) < 3 * np.e ** (epsilon_t) + 2:
        noise_result = []
        for value in values_t:
            noise_data = perturb_krr(domain, value, epsilon_t)
            noise_result.append(noise_data)
        category_count_noise = Counter(noise_result)
        cat2idx = {cat: i for i, cat in enumerate(domain)}
        for cat, cnt in category_count_noise.items():
            idx = cat2idx[cat]
            category_count_noise[idx] = cnt
        p = np.e ** (epsilon_t) / (np.e ** (epsilon_t) + len(domain) - 1)
        q = 1 / (np.e ** (epsilon_t) + len(domain) - 1)
        count_vector_noise = (np.array(category_count_noise) / len(values_t) - q) / (p - q)
    else:
        vectors = []
        for value in values_t:
            noise_data = perturb_oue(domain, value, epsilon_t)
            vectors.append(noise_data)
        count_vector_noise = np.sum(vectors, axis=0)
        p = 1 / 2
        q = 1 / (np.e ** (epsilon_t) + 1)
        count_vector_noise = (np.array(count_vector_noise) / len(values_t) - q) / (p - q)

    return count_vector_noise

def multi_user_perturbation(df, timestamp, epsilon, w, domain,column):

    all_result = []
    for i in tqdm(range(0, len(timestamp), w)):
        block_time = timestamp[i:i + w]
        epsilon_t = epsilon / len(block_time)
        for time in block_time:
            df_t = df[df['Time'] == time]
            values_t = df_t[column].values.tolist()
            count_vector_noise = perturb_value(values_t, domain, epsilon_t)
            all_result.append(count_vector_noise)

    return all_result


def mean_square_error_single(round, df, epsilon, w):

    # Volume traffic_volume
    # Unemployment LNU03000018
    # AirQualityUCI  C6H6(GT)
    column = 'LNU03000018'
    max_value, min_value, input_data = data_deal(df, column)
    real_mean = np.mean((np.array(df[column].values.tolist()) - min_value) / (max_value - min_value))
    # print(real_mean)
    MSE = 0
    for r in range(0, round):
         all_noise_result = single_user_perturbation(input_data, epsilon, w)
         noise_mean = np.mean((np.array(all_noise_result)+1)/2)
         # print(real_mean, noise_mean)
         mse = np.square(real_mean - noise_mean)
         print('第',r,'轮误差:',mse)
         MSE += mse

    return MSE/round

def mean_square_error_multi(round, df, epsilon, w):

    # Foursquare Country code
    column = 'Value'
    timestamp = df['Time'].unique().tolist()
    domain = df[column].unique().tolist()
    UID = df['User ID'].unique()
    # real_result = real_statistical_result(df, timestamp, domain)
    df_real = pd.read_json('statistical result/Syn2.json')
    real_result = df_real['real frequency'].values.tolist()
    MSE = 0
    for r in range(0, round):
        mse = 0
        noise_result = multi_user_perturbation(df, timestamp, epsilon, w, domain, column)
        for t in range(0,len(timestamp)):
            real = np.array(real_result[t])/len(UID)
            noise = np.array(noise_result[t])
            mse_t = np.mean(np.square(real - noise) )
            # print(t,':',mse_t)
            mse += mse_t
        print('第',r,'轮误差:', mse/len(timestamp))
        MSE += (mse/len(timestamp))

    return MSE / round


# range query
def generate_fixed_length_queries(domain_min, domain_max, query_len=10, query_num=10, seed=None):
    """
    随机生成 query_num 个长度相同的范围查询。
    query_len 表示区间跨度:
    query_len = 10 表示 [0,9], [2,11] 这种闭区间
    """
    if seed is not None:
        random.seed(seed)

    queries = []
    max_start = domain_max - query_len

    for _ in range(query_num):
        x = random.randint(domain_min, max_start)
        y = x + query_len
        queries.append((x, y))

    return queries

def mae_rangequery_multi(df, epsilon, w):

    # Foursquare Country code
    # syn2 and syn3 Value
    column = 'Value'
    timestamp = df['Time'].unique().tolist()
    domain = sorted(df[column].unique().tolist())
    UID = df['User ID'].unique()
    domain_min, domain_max = min(domain), max(domain)
    # domain_min, domain_max = 0, len(domain)
    df_real = pd.read_json('statistical result/Syn3.json')
    real_result = df_real['real frequency'].values.tolist()
    MAE = 0
    noise_result = multi_user_perturbation(df, timestamp, epsilon, w, domain, column)
    queries = generate_fixed_length_queries(domain_min, domain_max, query_len=10, query_num=1000, seed=None)
    for q in tqdm(queries):
        mae = 0
        for t in range(0,len(timestamp)):
            real_q = real_result[t][q[0]:q[1]]
            noise_q = noise_result[t][q[0]:q[1]]
            real = np.array(list(real_q))/len(UID)
            noise = np.array(noise_q)
            mae_t = np.mean(np.absolute(real - noise) )
            mae += mae_t
        MAE += (mae/len(timestamp))
    return MAE/len(queries)

if __name__ == "__main__":

    # Volume traffic_volume
    # Unemployment LNU03000018
    # AirQualityUCI  C6H6(GT)
    # Synthetic1 Value
    # Foursquare Country code
    # Synthetic2 Synthetic3 Value
    # TDrive_new LID

    Path = 'E:\\Code\\Stream data release LDP\\dataset\\Syn datasets\\'
    df = pd.read_csv(Path + 'Synthetic3.csv', encoding='unicode_escape')
    epsilon_list = [0.5, 1, 1.5, 2, 2.5, 3]
    # w_list = [60, 80, 100, 120]

    # range query
    for epsilon in epsilon_list:
        print('epsilon', epsilon)
        MAE = mae_rangequery_multi(df, epsilon, 80)
        print('mean absolute error', MAE)





