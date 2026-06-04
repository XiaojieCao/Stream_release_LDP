import numpy as np
import random
import krr,oue
import pandas as pd
from collections import Counter
from tqdm import tqdm

def perturb_krr(domain, input_value, epsilon):

    KRR = krr.KRR(domain, epsilon)
    perturbed_value = KRR.encode(tuple(input_value))
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

def data_deal(df,column):
    """
    syn1  max_value, min_value = 1000, 1
    Syn1, Volumn,
    """
    max_value = df[column].max()
    min_value = df[column].min()
    # max_value, min_value = 1, 1000
    df_new = df.copy()
    df_new[column] = df_new[column].apply(lambda x: 2*(x - min_value) / (max_value - min_value) - 1 )
    input_data = df_new[column].values.tolist()
    return max_value, min_value, input_data

def perturb_num(data, epsilon, w):
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
        noise_data = Piecewise_mechanism(value, epsilon / w)
        noise_result.append(noise_data)
    return noise_result

def group_single(data,epsilon, w):

    noise_result = perturb_num(data, epsilon, w)
    partitions = []

    # 记录已添加噪音数据的分组情况，用于计算dev
    G_noise = []

    # 记录当前数据分组情况
    G = []
    # 当前数据分区索引
    idx_G = []
    for t in range(len(data)):
        if t == 0:
            noise_data = noise_result[t]
            G_noise = G_noise + [noise_data]

            G = G + [noise_data]
            G_close = False
            idx_G.append(t)
        else:
            noise_data = noise_result[t]
            if G_close == True:
                G_noise = G_noise + [noise_result[t]]

                G = G + [noise_data]
                G_close = False
                idx_G = []
                idx_G.append(t)
            else:
                var_1 = pm_var(epsilon / w)
                group_values = G_noise.copy()
                group_values += [noise_result[t]]

                group_values = np.array(group_values)
                mean_val = np.mean(group_values)
                dev = np.sum(np.square(group_values - mean_val))- (len(group_values)+1/len(group_values))*var_1
                if dev < var_1:
                    G_noise = G_noise + [noise_result[t]]

                    G = G + [noise_data]
                    G_close = False
                    idx_G.append(t)
                else:
                    G = G_noise = []
                    G_close = True
                    idx_G = [t]

        current_partition = []
        if idx_G:
            current_partition = idx_G.copy()
        partitions.append(current_partition)
    return noise_result, partitions

#  multi-user
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
        for cat, cnt in category_count_noise.items():
            idx = cat2idx[cat]
            category_count_noise[idx] = cnt
        p = np.e ** (epsilon_t) / (np.e ** (epsilon_t) + len(domain) - 1)
        q = 1 / (np.e ** (epsilon_t) + len(domain) - 1)
        count_vector_noise = np.array(category_count_noise) / len(values_t) - q / (p - q)
    else:
        vectors = []
        for value in values_t:
            noise_data = perturb_oue(domain, value, epsilon_t)
            vectors.append(noise_data)
        count_vector_noise = np.sum(vectors, axis=0)
        p = 1 / 2
        q = 1 / (np.e ** (epsilon_t) + 1)
        count_vector_noise = (np.array(count_vector_noise) / len(values_t) - q) / (p - q)
    estimates = norm(count_vector_noise)
    return estimates

def perturb_categorical(df, timestamp, domain, epsilon, w, column):
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
        count_vector_noise = perturb_value(values_t,domain, epsilon/w)
        estimates = norm(count_vector_noise)
        all_noise_result.append(estimates)
    return all_noise_result

def group_multi(df,timestamp, domain, epsilon, w, column):

    noise_result = perturb_categorical(df, timestamp, domain, epsilon, w, column)
    partitions = []
    # 记录已添加噪音数据的分组情况，用于计算dev
    G_noise = np.empty((0, len(domain)))

    # 记录当前数据分组情况
    G = []
    # 当前数据分区索引
    idx_G = []
    for t in tqdm(range(len(timestamp))):
        time = timestamp[t]
        df_t = df[df['Time'] == time]
        values_t = df_t[column].values.tolist()
        if t == 0:
            count_vector_noise = noise_result[t]
            G_noise = np.vstack([G_noise, np.array(count_vector_noise)])

            G = G + [count_vector_noise]
            G_close = False
            idx_G.append(t)
        else:
            count_vector_noise = noise_result[t]
            if G_close == True:
                G_noise = np.vstack([G_noise, np.array(count_vector_noise)])

                G = G + [count_vector_noise]
                G_close = False
                idx_G = []
                idx_G.append(t)
            else:
                var, var_extra = Var(domain, epsilon / w, len(df_t))
                group_values = G_noise.copy()
                group_values = np.vstack([group_values, np.array(count_vector_noise)])

                group_values = np.array(group_values)
                mean = np.mean(group_values, axis=0)
                len_G = group_values.shape[0]
                dev = np.sum((group_values - mean)**2)- (len_G+1/len_G)*(var + var_extra)*len(domain)
                if dev < (var + var_extra)*len(domain):
                    G_noise = np.vstack([G_noise, np.array(count_vector_noise)])

                    G = G + [count_vector_noise]
                    G_close = False
                    idx_G.append(t)
                else:
                    G_noise = np.empty((0, len(domain)))
                    G = []
                    G_close = True
                    idx_G = [t]

        current_partition = []
        if idx_G:
            current_partition = idx_G.copy()
        partitions.append(current_partition)

    return noise_result, partitions

def Smoother(noise_result, partitions, flag):

    all_noise_result = []
    for t in range(len(noise_result)):
        Pt = partitions[t]
        group_values = [noise_result[i] for i in Pt]
        if flag == 0:
                avg = np.mean(group_values)
                all_noise_result.append(avg)
        elif flag == 1:
                med = np.median(group_values)
                all_noise_result.append(med)
        elif flag == 2:
                avg = np.mean(group_values)
                estimate = (noise_result[t] - avg) / len(group_values) + avg
                all_noise_result.append(estimate)
    return all_noise_result

def Smoother_multi(noise_result, partitions):

    all_noise_result = []
    for t in range(len(noise_result)):
        Pt = partitions[t]
        group_values = np.array([noise_result[i] for i in Pt])
        avg = np.mean(group_values, axis=0)
        all_noise_result.append(avg)
    return all_noise_result

# single-user
def mean_square_error_single(round, df, epsilon, w):

    # Volume traffic_volume
    # Unemployment LNU03000018
    # AirQualityUCI  C6H6(GT)
    # Syn1 Value
    column = 'traffic_volume'
    max_value, min_value, input_data = data_deal(df, column)
    real_mean = np.mean(np.array(df[column].values.tolist()) - min_value) / (max_value - min_value)

    MSE = 0
    for r in tqdm(range(0, round)):
         noise_result,partitions = group_single(input_data, epsilon, w)
         all_noise_result = Smoother(noise_result, partitions, 0)
         noise_mean = np.mean((np.array(all_noise_result)+1)/2)
         # print(real_mean, noise_mean)
         mse = np.square(real_mean - noise_mean)
         print('第',r,'轮误差:',mse)
         MSE += mse
    return MSE / round

# multi-user
def mean_square_error_multi(round, df, epsilon, w):

    # Foursquare Country code
    # syn2 and syn3 Value
    column = 'Value'
    timestamp = df['Time'].unique().tolist()
    domain = df[column].unique().tolist()
    UID = df['User ID'].unique()
    df_real = pd.read_json('statistical result/Syn3_n_80w.json')
    real_result = df_real['real frequency'].values.tolist()

    MSE = 0
    for r in range(0, round):
        mse = 0
        noise_result,partitions = group_multi(df,timestamp, domain, epsilon, w, column)
        noise_result = Smoother_multi(noise_result, partitions)
        for t in tqdm(range(0,len(timestamp))):
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
    domain_min, domain_max = min(domain), max(domain)
    # domain_min, domain_max = 0, len(domain)
    df_real = pd.read_json('statistical result/Syn3.json')
    real_result = df_real['real frequency'].values.tolist()
    noise_result, partitions = group_multi(df, timestamp, domain, epsilon, w, column)
    noise_result = Smoother_multi(noise_result, partitions)

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
        print(mae/len(timestamp))
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

    # Path = 'E:\\Code\\Stream data release LDP\\dataset\\'
    # df = pd.read_csv(Path + 'Unemployment.csv', encoding='unicode_escape')
    # epsilon_list = [0.5, 1, 1.5, 2, 2.5, 3]
    # w_list = [60, 80, 100, 120]
    # for epsilon in epsilon_list:
    #     print('epsilon:', epsilon)
    # for w in w_list:
    #     print('w:', w)
    #     MSE = mean_square_error_multi(5, df, 2, w)
    #     print('mean_relative_error:', MSE)

    # range query
    epsilon_list = [ 2.5, 3]
    for epsilon in epsilon_list:
        print('epsilon', epsilon)
        MAE = mae_rangequery_multi(df, epsilon, 80)
        print('mean absolute error', MAE)

