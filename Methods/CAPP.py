import numpy as np
import pandas as pd
import math
import random


def square_wave_mechanism(t, epsilon):
    # input value : t
    b = (epsilon * np.exp(epsilon) - np.exp(epsilon) + 1) / (
            2 * np.exp(epsilon) * (np.exp(epsilon) - 1 - epsilon))
    x = random.uniform(0, 1)
    if x < (2 * b * np.e ** epsilon) / (2 * b * np.e ** epsilon + 1):
        perturbed_t = random.uniform(- b + t, b + t)
    else:
        x_1 = random.uniform(0, 1)
        if x_1 <= t:
            perturbed_t = random.uniform(- b, - b + t)
        else:
            perturbed_t = random.uniform(b + t, b + 1)
    perturbed_t = (perturbed_t + b) / (2 * b + 1)
    return perturbed_t


def Simple_Moving_Average(noise_result, k):
    smooth_result = []
    for t in range(len(noise_result)):
        # 确定窗口边界
        left = max(0, t - k)
        right = min(len(noise_result) - 1, t + k)
        # 取出2k+1个窗口内的数据
        window = noise_result[left:right + 1]
        # 计算均值
        avg_value = np.mean(window)

        smooth_result.append(avg_value)

    return smooth_result


def compute_l_u(epsilon, value):
    b = (epsilon * (np.e ** (epsilon)) - np.e ** epsilon + 1) / (
            2 * np.e ** (epsilon) * (np.e ** epsilon - 1 - epsilon))
    p = np.e ** epsilon / (2 * b * np.e ** epsilon + 1)
    q = 1 / (2 * b * np.e ** epsilon + 1)

    Dx = q * ((1 + 2 * b) * value - (b + 1 / 2))
    es = math.exp(1-Dx) - 1

    var = (2 * p * (b ** 3)) / 3 - np.square(b * q) + q * np.square(b) - b * np.square(q) + b * q - np.square(q) / 4 + (q / 3)
    ed = np.sqrt(var)

    T = es - ed
    l = 0 - T
    u = 1 + T

    return l, u


# w windows
def CAPP(data, epsilon):

    D = 0
    noise_result = []
    for value in data:
        l, u = compute_l_u(epsilon, value)

        value_new = value + D
        if value_new < l:
            value_new = l
        elif value_new > u:
            value_new = u
        else:
            value_new = value_new
        # normalization
        nor_value = (value_new - l) / (u - l)
        #perturbation
        noise_value = square_wave_mechanism(nor_value, epsilon)
        #dennormalization
        noise_value = noise_value * (u - l) + l

        noise_result.append(noise_value)
        # Calculate deviation
        dev_i = value - noise_value
        D = D + dev_i

    return noise_result

def output_noise_result(data,epsilon, w, k):

    all_noise_result = []
    for i in range(0, len(data), w):
        data_block = data[i:i + w]
        epsilon_w = epsilon / w
        noise_result = CAPP(data_block, epsilon_w)
        smooth_result = simple_moving_average(noise_result, k)
        all_noise_result.append(list(smooth_result))
    all_noise_result = sum(all_noise_result, [])

    return all_noise_result

def simple_moving_average(data, k):
    """
    参数：
        data: input noise data stream
        k: int
    平滑窗口半径，窗口大小为 2k + 1
    """
    k = int(k)
    n = len(data)
    smoothed = np.zeros(n)
    window_size = 2 * k + 1
    for t in range(n):
        # 确定窗口范围
        start = max(0, t - k)
        end = min(n, t + k + 1)
        # 平均可用数据
        smoothed[t] = np.mean(data[start:end])
    return smoothed

# Volume dataset: data processing
def data_deal(df,column):

    max_value = df[column].max()
    min_value = df[column].min()
    # max_value, min_value = 1, 1000
    df_new = df.copy()
    df_new[column] = df_new[column].apply(lambda x: (x - min_value) / (max_value - min_value))
    input_data = df_new[column].values.tolist()

    return input_data

def mean_square_error_single(round, df, epsilon, w, k):

    # Volume traffic_volume
    # Unemployment LNU03000018
    # AirQualityUCI  C6H6(GT)
    # Synthetic1 Value
    column = 'traffic_volume'
    input_data = data_deal(df, column)
    real_mean = np.mean(input_data)

    MSE = 0
    for r in range(0, round):
         all_noise_result = output_noise_result(input_data, epsilon, w, k)
         noise_mean = np.mean(all_noise_result)
         # print(real_mean, noise_mean)
         mse = np.square(real_mean - noise_mean)
         print('第',r,'轮误差:',mse)
         MSE += mse
    return MSE / round


if __name__ == "__main__":

    # Volume traffic_volume
    # Unemployment LNU03000018
    # AirQualityUCI  C6H6(GT)
    # Synthetic1 Value
    # Foursquare Country code
    # Synthetic2 Synthetic3 Value
    # TDrive_new LID

    Path = 'E:\\Code\\Stream data release LDP\\dataset\\'
    df = pd.read_csv(Path + 'Unemployment.csv', encoding='unicode_escape')
    # epsilon_list = [0.5, 1, 1.5, 2, 2.5, 3]
    w_list = [60, 80, 100, 120]
    #
    # for epsilon in epsilon_list:
    #     print('epsilon:', epsilon)
    for w in w_list:
        print('w:', w)
        MSE = mean_square_error_single(10, df, 2, w, w/5)
        print('mean square error:',MSE)
