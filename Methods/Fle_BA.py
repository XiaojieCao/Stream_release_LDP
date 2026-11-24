import numpy as np
import random
import krr,oue
import pandas as pd
from collections import Counter
from tqdm import tqdm

def perturb_krr(domain, input_value, epsilon):

    KRR = krr.KRR(domain, epsilon)
    perturbed_value = KRR.encode(input_value)

    return perturbed_value

def perturb_oue(domain, input_value, epsilon):

    OUE = oue.OUE(len(domain), epsilon)
    index = domain.index(input_value)
    perturbed_value = OUE.perturb(index)

    return perturbed_value

def Var(domain, e, N):
    """
    Compute the variance and extra variance of frequency estimation
    based on the domain size and privacy budget.
    Parameters:
        domain (list): d is the domain size
        e (float): Privacy budget
        N (int): Number of users
    Returns:
        tuple: (var, var_extra)
    """
    d = len(domain)
    if d < (3 * np.exp(e) + 2):
        # Variance for GRR (variance of  frequency)
        var = (1/N) * ((d - 2) + np.exp(e)) / ((np.exp(e) - 1) ** 2)
        var_extra = (1 / d) * (1/N) * (d - 2) / (np.exp(e) - 1)
    else:
        # Variance for OUE
        var = (1/N) * (4 * np.exp(e)) / ((np.exp(e) - 1) ** 2)
        var_extra = (1 / d) * (1/N)

    return var, var_extra

def Piecewise_mechanism(t, epsilon):
    C = (np.exp(epsilon / 2) + 1) / (np.exp(epsilon / 2) - 1)
    p = (np.exp(epsilon) - np.exp(epsilon / 2)) / (2 * np.exp(epsilon / 2) + 2)
    l = (C + 1) / 2 * t - (C - 1) / 2
    r = l + C - 1
    p_h = (p - p / np.exp(epsilon)) * (C - 1)
    rnd = np.random.random()
    if rnd <= p_h:
        perturbed_t = random.uniform(l, r)
    else:
        perturbed_t = random.uniform(- C, C)
    perturbed_t = perturbed_t / C

    return perturbed_t

def pm_var(epsilon):
    var = 4*np.exp(epsilon/2) / 3*np.square(np.exp(epsilon/2) - 1)

    return var


def perturb_M1_num(data, epsilon_1, w):
    """
    Parameters
    ----------
    data : input data
    epsilon_1 : Private dissimilarity calculation
    w : window size
    Returns :  noise data
    """
    noise_result = []
    for value in data:
        noise_data = Piecewise_mechanism(value, epsilon_1/w)
        noise_result.append(noise_data)

    return noise_result

# Volume dataset: data processing
# traffic_volume data type : Integer
def data_deal(df,column):

    # Syn1 min(1) max(1000)
    max_value = df[column].max()
    min_value = df[column].min()
    # max_value, min_value = 1, 1000
    df_new = df.copy()
    df_new[column] = df_new[column].apply(lambda x: 2*(x - min_value) / (max_value - min_value) - 1 )
    input_data = df_new[column].values.tolist()

    return max_value,min_value,input_data

# Privacy Budget Absorption
def LDP_PBA_Single(data,epsilon_1, epsilon_2, w):

    noise_result_1 = perturb_M1_num(data, epsilon_1, w)
    noise_result_2 = []

    used_eps = []
    unused_eps = []
    for t in range(len(data)):
        value = data[t]
        if t == 0 :
            epsilon_t = epsilon_2 / w
            noise_data = Piecewise_mechanism(value, epsilon_t)

            noise_result_2.append(noise_data)
            used_eps.append(epsilon_t)
        else:
            if sum(used_eps) == epsilon_2:
                noise_result_2.append(noise_result_2[t - 1])
                used_eps.append(0)
            else:
                if len(unused_eps) == w:
                    del unused_eps[0]
                epsilon_t = sum(unused_eps) + (epsilon_2 / w)
                # 计算dis
                var_1 = pm_var(epsilon_1 / w)
                cur_data = noise_result_1[t]
                pre_data = noise_result_2[t - 1]
                dis = (cur_data - pre_data) ** 2 - var_1

                err = pm_var(epsilon_t)
                if dis > err:
                    noise_data = Piecewise_mechanism(value, epsilon_t)

                    noise_result_2.append(noise_data)
                    used_eps.append(epsilon_t)
                    unused_eps.clear()
                else:
                    noise_result_2.append(noise_result_2[t - 1])
                    used_eps.append(0)
                    unused_eps.append(epsilon_2 / w)
            if len(used_eps) == w:
                    del used_eps[0]

    return noise_result_2

def norm(count_vector_noise):
    estimates = np.copy(count_vector_noise)
    total = sum(estimates)
    sum_p = 1
    domain_size = len(count_vector_noise)
    diff = (sum_p - total) / domain_size
    estimates += diff

    return estimates

def perturb_value(values_t,domain, epsilon_t):
    user_num = len(values_t)
    domain_size = len(domain)
    exp_epsilon = np.exp(epsilon_t)

    if domain_size < 3 * exp_epsilon + 2:
        noise_result = []
        for value in values_t:
            noise_data = perturb_krr(domain, value, epsilon_t)
            noise_result.append(noise_data)

        category_count_noise = Counter(noise_result)
        cat2idx = {cat: i for i, cat in enumerate(domain)}

        # 用独立数组存储计数，没有出现的类别保持为 0。
        count_vector_noise = np.zeros(domain_size, dtype=float)

        for cat, cnt in category_count_noise.items():
            count_vector_noise[cat2idx[cat]] = cnt

        p = exp_epsilon / (exp_epsilon + domain_size - 1)
        q = 1 / (exp_epsilon + domain_size - 1)

    else:
        vectors = []
        for value in values_t:
            noise_data = perturb_oue(domain, value, epsilon_t)
            vectors.append(noise_data)

        count_vector_noise = np.sum(vectors, axis=0)

        p = 0.5
        q = 1 / (exp_epsilon + 1)

    # 两种机制均进行去偏，得到频率估计。
    count_vector_noise = (
            (count_vector_noise / user_num - q) / (p - q)
    )

    return norm(count_vector_noise)

def perturb_M1_categorical(df, timestamp, domain, epsilon_1, w, column):
    """
    Parameters
    timestamp : time series
    epsilon_1 : Private dissimilarity calculation
    w : window size
    Returns :  noise data
    """
    all_noise_result = []
    for t in tqdm(range(len(timestamp))):
        time = timestamp[t]
        df_t = df[df['Time'] == time]
        values_t = df_t[column].values.tolist()
        count_vector_noise = perturb_value(values_t, domain, epsilon_1/w)
        estimates = norm(count_vector_noise)
        all_noise_result.append(estimates)

    return all_noise_result

def LDP_PBA_multi(df, timestamp,epsilon_1, epsilon_2, domain, w, column):

    noise_result_1 = perturb_M1_categorical(df, timestamp, domain, epsilon_1, w, column)
    noise_result_2 = []

    used_eps = []
    unused_eps = []
    for t in tqdm(range(0,len(timestamp))):
        time = timestamp[t]
        if t == 0 :
            epsilon_t = epsilon_2 / w
            df_t = df[df['Time'] == time]
            values_t = df_t[column].values.tolist()
            count_vector_noise = perturb_value(values_t, domain, epsilon_t)

            noise_result_2.append(count_vector_noise)
            used_eps.append(epsilon_t)
        else:
            df_t = df[df['Time'] == time]
            values_t = df_t[column].values.tolist()
            if sum(used_eps) == epsilon_2:
                noise_result_2.append(noise_result_2[t - 1])
                used_eps.append(0)
            else:
                if len(unused_eps) == w:
                    del unused_eps[0]
                epsilon_t = sum(unused_eps) + (epsilon_2 / w)
                # 计算dis
                var, var_extra = Var(domain, epsilon_2 / w, len(df_t))
                cur_data = noise_result_1[t]
                pre_data = noise_result_2[t - 1]
                dis = np.sum(np.square(np.array(cur_data) - np.array(pre_data))) / len(domain) - (var + var_extra)

                var_t, var_extra_t = Var(domain, epsilon_t, len(df_t))
                err = var_t + var_extra_t
                if dis > err:
                    count_vector_noise = perturb_value(values_t, domain, epsilon_t)
                    noise_result_2.append(count_vector_noise)
                    used_eps.append(epsilon_t)
                    unused_eps.clear()
                else:
                    noise_result_2.append(noise_result_2[t - 1])
                    used_eps.append(0)
                    unused_eps.append(epsilon_2 / w)
            if len(used_eps) == w:
                    del used_eps[0]

    return noise_result_2

def mean_square_error_single(round, df, epsilon, w):

    # Volume traffic_volume
    # Unemployment LNU03000018
    # AirQualityUCI  C6H6(GT)
    column = 'traffic_volume'
    max_value, min_value, input_data = data_deal(df, column)
    real_mean = np.mean((np.array(df[column].values.tolist()) - min_value) / (max_value - min_value))
    # print(real_mean)

    epsilon_1 = epsilon_2 = epsilon / 2
    MSE = 0
    for r in range(0, round):
         all_noise_result = LDP_PBA_Single(input_data,epsilon_1, epsilon_2, w)
         noise_mean = np.mean((np.array(all_noise_result)+1)/2)
         # print(real_mean, noise_mean)
         mse = np.square(real_mean - noise_mean)
         print('第',r,'轮误差:',mse)
         MSE += mse

    return MSE/round

def mean_square_error_multi(round, df, epsilon, w):

    # Foursquare Country code
    # syn2 and syn3 Value
    column = 'Value'
    timestamp = df['Time'].unique().tolist()
    domain = df[column].unique().tolist()
    UID = df['User ID'].unique()
    df_real = pd.read_json('statistical result/Syn2_n_80w.json')
    real_result = df_real['real frequency'].values.tolist()

    epsilon_1 = 0.5*epsilon
    epsilon_2 = 0.5*epsilon
    MSE = 0
    for r in range(0, round):
        mse = 0
        noise_result = LDP_PBA_multi(df, timestamp,epsilon_1, epsilon_2, domain, w, column)
        for t in tqdm(range(0, len(timestamp))):
            real = np.array(real_result[t])/len(UID)
            noise = np.array(noise_result[t])
            mse_t = np.mean(np.square(real - noise) )
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
    # domain_min, domain_max = min(domain), max(domain)
    domain_min, domain_max = 0, len(domain)
    df_real = pd.read_json('statistical result/Syn3.json')
    real_result = df_real['real frequency'].values.tolist()

    epsilon_1 = epsilon_2 = epsilon / 2
    noise_result = LDP_PBA_multi(df, timestamp,epsilon_1, epsilon_2, domain, w, column)

    MAE = 0
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
        # print(mae/len(timestamp))
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

    Path = 'E:\\Code\\Stream_data_release_LDP\\dataset\\Syn datasets\\'
    df = pd.read_csv(Path + 'Synthetic2_n_80w.csv', encoding='unicode_escape')
    epsilon_list = [2.5,3]
    w_list = [40,60,80,100,120]
    MSE = mean_square_error_multi(2, df, 2, 20)
    print('mean square error:', MSE)
    # for epsilon in epsilon_list:
    #     print('epsilon:', epsilon)
    # for w in w_list:
    #     print('w:', w)
    #     MSE = mean_square_error_multi(2, df, epsilon, 20)
    #     print('mean square error:', MSE)

    # for w in w_list:
    #     print('w:', w)
    #     MSE = mean_square_error_single(5, df, 2, w)
    #     print('mean square error:', MSE)

    # range query
    # for epsilon in epsilon_list:
    #     print('epsilon', epsilon)
    # for w in w_list:
    #     print('w', w)
    #     MAE = mae_rangequery_multi(df, 2, w)
    #     print('mean absolute error', MAE)


