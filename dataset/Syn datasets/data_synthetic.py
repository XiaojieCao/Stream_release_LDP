import numpy as np
import scipy.stats as ss
import pandas as pd

# Syn1, Syn2, and Syn3 datasets
# Syn2_d_X / Syn3_d_X: fix users n, vary domain d {200,400,800}
# Syn2_n_X / Syn3_n_X: fix domain d, vary users n {20w,40w,80w}

def syn2(num_users, T, d):

    value_range = (0, d-1)
    switch_point = T // 2  # 前半段/后半段分界点
    np.random.seed()
    # 生成用户ID
    user_ids = np.arange(1, num_users + 1)
    # 生成时间索引
    time_points = np.arange(1, T + 1)
    # 生成前半段（均匀分布）
    data_uniform = np.random.randint(value_range[0], value_range[1] + 1, size=(num_users, switch_point))

    x = np.arange(0, d)
    xU, xL = x + 0.5, x - 0.5
    prob = ss.norm.cdf(xU, loc= 90, scale=10) - ss.norm.cdf(xL, loc = 90, scale = 10)
    prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
    data_normal = np.random.choice(x, size=(num_users, T - switch_point), p=prob)
    # 拼接前后两部分
    data_full = np.hstack([data_uniform, data_normal])

    df = pd.DataFrame(data_full, index=user_ids, columns=[f"T{t}" for t in time_points])
    df.index.name = "User ID"
    df_long = df.reset_index().melt(id_vars="User ID", var_name="Time", value_name="Value")

    return df_long

def syn3(num_users, T, d):

    data = np.zeros((num_users, T), dtype=int)

    x = np.arange(0, d)
    # print(x)
    xU, xL = x + 0.5, x - 0.5
    for t in range(T):
        mean = np.random.uniform(0, d)
        var = np.random.uniform(0, 10)

        # 随机选择分布类型：0=正态，1=拉普拉斯
        dist_type = np.random.choice([0, 1])
        # 计算离散概率
        if dist_type == 0:
            prob = ss.norm.cdf(xU, loc=mean, scale=np.sqrt(var)) - ss.norm.cdf(xL, loc=mean, scale=np.sqrt(var))
        else:
            prob = ss.laplace.cdf(xU, loc=mean, scale=np.sqrt(var)) - ss.laplace.cdf(xL, loc=mean, scale=np.sqrt(var))
        prob = prob / prob.sum()  # 归一化
        # 按概率采样一个类别
        samples = np.random.choice(x, size=num_users, p=prob)
        data[:, t] = samples

    # new DataFrame
    df = pd.DataFrame(data, index=np.arange(1, num_users + 1), columns=[f"T{t + 1}" for t in range(T)])
    df.index.name = "User ID"

    # 转成长表形式（可选）
    df_long = df.reset_index().melt(id_vars="User ID", var_name="Time", value_name="Value")

    return df_long

def syn1(T, d):

    result = []
    for t in range(T):
        mean = np.random.uniform(0, d-1)
        var = np.random.uniform(0, 10)
        dist_type = np.random.choice([0, 1])
        # 计算离散概率
        if dist_type == 0:
            data = np.random.normal(loc=mean, scale=np.sqrt(var))
        else:
            data = np.random.laplace(loc=mean, scale=np.sqrt(var/2))
        data = np.clip(data, 1, d)
        result.append(data)

    # # 前半段：均匀分布 U(1, 1000)
    # first_half = np.random.uniform(low=1, high=1000, size=T//2)
    # # 后半段：正态分布 N(900, 10)
    # # 正态分布可能生成超出区间的值，可以 clip 控制在 [0,1000]
    # second_half = np.random.normal(loc=900, scale=np.sqrt(10), size=T - T//2)
    # second_half = np.clip(second_half, 1, 1000)
    # # 拼接
    # data = np.concatenate([first_half, second_half])

    time_stamps = np.arange(T)
    # 生成 DataFrame
    df = pd.DataFrame({
        "Time": time_stamps,
        "Value": result
    })
    return df



if __name__ == "__main__":
    num_users = 800000
    T = 500
    d = 100
    df_long = syn3(num_users, T, d)
    print(df_long)
    df_long.to_csv('Synthetic3_n_80w.csv', index=False)