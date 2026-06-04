import numpy as np
import pandas as pd
import math
import random
import oue,krr
from collections import Counter
from tqdm import tqdm


def split_users(n,w):
    groups = []
    group_size = math.floor(n/w)

    for i in range(w):
        start = i * group_size + 1
        end = (i+1) * group_size
        groups.append([start, end])

    return groups

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
    perturbed_t = perturbed_t / C
    return perturbed_t

def perturb_krr(domain, input_value, epsilon):

    KRR = krr.KRR(domain, epsilon)
    # perturbed_value = KRR.encode(tuple(input_value))
    perturbed_value = KRR.encode(input_value)
    return perturbed_value

def perturb_oue(domain, input_value, epsilon):

    OUE = oue.OUE(len(domain), epsilon)
    index = domain.index(input_value)
    perturbed_value = OUE.perturb(index)
    return perturbed_value

# single user
# data processing
def data_deal(df,column):

    # Syn1 min(1) max(1000)
    max_value = df[column].max()
    min_value = df[column].min()
    # max_value, min_value = 1, 1000
    df_new = df.copy()
    df_new[column] = df_new[column].apply(lambda x: 2*(x - min_value) / (max_value - min_value) -1)
    input_data = df_new[column].values.tolist()
    return max_value, min_value, input_data

def sliding_window_k_sample(data, epsilon, w, k):
    """
    滑动窗口采样机制：在任意长度为 w 的窗口内，采样 k 个数据。
    索引0必须被采样。
    未采样的位置复用最近采样的扰动值。

    参数：
    data: 原始数据列表
    epsilon: 隐私预算
    w: 窗口长度
    k: 每个窗口允许的采样数量
    input_data_max: 数据最大值，用于扰动
    """
    timestamp = len(data)
    all_noise_result = []
    sampled_indices = set()  # 已采样位置
    last_noise_value = None

    for t in range(0, timestamp):
        value = data[t]
        if t == 0:
            # 初始第0个位置必须被采样
            first_noise = Piecewise_mechanism(value, epsilon/k)
            last_noise_value = first_noise
            sampled_indices.add(0)
        else:
            # 当前滑动窗口范围
            window_start = max(0, t - w + 1)
            window_end = t

            # 当前窗口内已采样数量
            current_window_samples = [i for i in sampled_indices if window_start <= i <= window_end]
            sampled_count = len(current_window_samples)

            # 若当前窗口采样数 < k，则可能采样当前点
            if sampled_count < k:
                # 还需要多少个采样点
                need = k - sampled_count

                # 剩余可用位置数
                remaining = window_start + w - 1 - t
                remaining = max(remaining, 0)

                # 若剩余位置数 == 需要采样数，则强制采样
                if remaining <= need - 1:
                    do_sample = True
                else:
                    # 否则按随机决定
                    do_sample = bool(random.getrandbits(1))
            else:
                do_sample = False  # 当前窗口已满，不再采样

            if do_sample:
                noise_value = Piecewise_mechanism(value, epsilon/k)
                last_noise_value = noise_value
                sampled_indices.add(t)
            # 未采样则复用上次噪声
        all_noise_result.append(last_noise_value)

    return all_noise_result

# multi user
def norm(count_vector_noise):
    estimates = np.copy(count_vector_noise)
    total = sum(estimates)
    sum_p = 1
    domain_size = len(count_vector_noise)
    diff = (sum_p - total) / domain_size
    estimates += diff
    return estimates

def perturb_value(values_t,domain, epsilon_t):

    if len(domain) < 3 * np.e ** (epsilon_t) + 2:
        noise_result = []
        for value in values_t:
            noise_data = perturb_krr(domain, value, epsilon_t)
            noise_result.append(noise_data)
        category_count_noise = Counter(noise_result)
        cat2idx = {cat: i for i, cat in enumerate(domain)}
        new_counter = Counter()
        for cat, cnt in category_count_noise.items():
            idx = cat2idx[cat]
            new_counter[idx] = cnt
        category_count_noise = new_counter

        count_vector = np.zeros(len(domain))
        for idx, cnt in category_count_noise.items():
            count_vector[idx] = cnt
        p = np.e ** epsilon_t / (np.e ** epsilon_t + len(domain) - 1)
        q = 1 / (np.e ** epsilon_t + len(domain) - 1)
        count_vector_noise = (count_vector  / len(values_t) - q) / (p - q)
    else:
        vectors = []
        for value in values_t:
            noise_data = perturb_oue(domain, value, epsilon_t)
            vectors.append(noise_data)
        count_vector_noise = np.sum(vectors, axis=0)
        p = 1 / 2
        q = 1 / (np.e ** epsilon_t + 1)
        count_vector_noise = (np.array(count_vector_noise)/len(values_t) - q )/ (p - q)
        # count_vector_noise = count_vector_noise / np.sum(count_vector_noise)
    estimates = norm(count_vector_noise)
    return estimates

def sliding_window_k_sample_multi(df, timestamp, epsilon, w, domain, k, column):

    all_noise_result = []
    sampled_indices = set()  # 已采样位置
    last_noise_value = None
    for t in tqdm(range(0, len(timestamp))):
        time = timestamp[t]
        if t == 0:
            # 初始第0个位置必须被采样
            df_t = df[df['Time'] == time]
            values_t = df_t[column].values.tolist()
            count_vector_noise = perturb_value(values_t, domain, epsilon / k)
            last_noise_value = count_vector_noise
            sampled_indices.add(0)
        else:
            # 当前滑动窗口范围
            window_start = max(0, t - w + 1)
            window_end = t
            # 当前窗口内已采样数量
            current_window_samples = [i for i in sampled_indices if window_start <= i <= window_end]
            sampled_count = len(current_window_samples)
            # 若当前窗口采样数 < k，则可能采样当前点
            if sampled_count < k:
                # 还需要多少个采样点
                need = k - sampled_count
                # 剩余可用位置数
                remaining = window_start + w - 1 - t
                remaining = max(remaining, 0)
                # 若剩余位置数 == 需要采样数，则强制采样
                if remaining <= need - 1:
                    do_sample = True
                else:
                    do_sample = bool(random.getrandbits(1))  # 否则按随机决定
            else:
                do_sample = False  # 当前窗口已满，不再采样

            if do_sample:
                df_t = df[df['Time'] == time]
                values_t = df_t[column].values.tolist()
                count_vector_noise = perturb_value(values_t, domain, epsilon / k)
                last_noise_value = count_vector_noise
                sampled_indices.add(t)
            # 未采样则复用上次噪声
        all_noise_result.append(last_noise_value)
    return all_noise_result

def mean_square_error_single(round, df, epsilon, w,k):
    # Volume traffic_volume
    # Unemployment LNU03000018
    # AirQualityUCI  C6H6(GT)
    # Synthetic1 Value
    column = 'traffic_volume'
    max_value, min_value, input_data = data_deal(df, column)
    real_mean = np.mean(np.array(df[column].values.tolist()) - min_value) / (max_value - min_value)

    MSE = 0
    for r in range(0, round):
         all_noise_result = sliding_window_k_sample(input_data,epsilon, w, k)
         # print(all_noise_result)
         noise_mean = np.mean((np.array(all_noise_result)+1)/2)
         # print(noise_mean)
         mse = np.square(real_mean - noise_mean)
         print('第',r,'轮误差:',mse)
         MSE += mse
    return MSE / round

def mean_square_error_multi(round, df, epsilon, w, k):

    # Foursquare Country code
    # syn2 and syn3 Value
    column = 'Value'
    timestamp = df['Time'].unique().tolist()
    domain = df[column].unique().tolist()
    UID = df['User ID'].unique()
    df_real = pd.read_json('statistical result/Syn2.json')
    real_result = df_real['real frequency'].values.tolist()
    # print(real_result)
    MSE = 0
    for r in range(0, round):
        mse = 0
        noise_result = sliding_window_k_sample_multi(df, timestamp, epsilon, w, domain, k, column)
        for t in tqdm(range(0,len(timestamp))):
            real = np.array(real_result[t])/len(UID)
            noise = np.array(noise_result[t])
            mse_t = np.mean(np.square(real - noise) )
            mse += mse_t
        print('第',r,'轮误差:', mse/len(timestamp))
        MSE += (mse/len(timestamp))
    return MSE / round

def MAE_single(round, df, epsilon, w,k):
    # Volume traffic_volume
    # Unemployment LNU03000018
    # AirQualityUCI  C6H6(GT)
    # Synthetic1 Value
    column = 'Value'
    max_value, min_value, input_data = data_deal(df, column)
    real_mean = np.mean(np.array(df[column].values.tolist()) - min_value) / (max_value - min_value)

    MAE = 0
    for r in range(0, round):
         all_noise_result = sliding_window_k_sample(input_data,epsilon, w, k)
         # print(all_noise_result)
         noise_mean = np.mean((np.array(all_noise_result)+1)/2)
         # print(noise_mean)
         mae = np.absolute(real_mean - noise_mean)
         print('第',r,'轮误差:',mae)
         MAE += mae
    return MAE / round

def MAE_multi(round, df, epsilon, w, k):

    # Foursquare Country code
    # syn2 and syn3 Value
    column = 'Country code'
    timestamp = df['Time'].unique().tolist()
    domain = df[column].unique().tolist()
    UID = df['User ID'].unique()
    df_real = pd.read_json('statistical result/Four.json')
    real_result = df_real['real frequency'].values.tolist()
    # print(real_result)
    MAE = 0
    for r in range(0, round):
        mae = 0
        noise_result = sliding_window_k_sample_multi(df, timestamp, epsilon, w, domain, k, column)
        for t in tqdm(range(0,len(timestamp))):
            real = np.array(real_result[t])/len(UID)
            noise = np.array(noise_result[t])
            mae_t = np.mean(np.absolute(real - noise))
            mae += mae_t
        print('第',r,'轮误差:', mae/len(timestamp))
        MAE += (mae/len(timestamp))
    return MAE / round


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


def mae_rangequery_multi(df, epsilon, w, k):

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
    noise_result = sliding_window_k_sample_multi(df, timestamp, epsilon, w, domain, k, column)
    queries = generate_fixed_length_queries(domain_min, domain_max, query_len=10, query_num=1000, seed=None)
    # print(queries)
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
        MAE = mae_rangequery_multi(df, epsilon, 80, 20)
        print('mean absolute error', MAE)

