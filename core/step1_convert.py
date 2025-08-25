import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler

# 性别编码函数：将'M'映射为1，其他映射为0
def M_1_F_0(instance):
    if(instance=='M'):
        return 1;
    return 0;

# 根据交易时间的小时数划分时间段（夜间 or 白天）
def time_category(hour):
    if hour>=22 or hour<=3 :
        return "night";
    else :
        return "day"

# 根据交易时间是周几，划分为两个类别（周末+周六日：cat1，工作日：cat2）
def category_day(day):
    if(day==0 or day>=5):
        return "cat1";
    else:
        return "cat2"

def preprocess_creditcard_data(df):
    """
    对信用卡交易数据进行预处理和特征工程：
    - 时间解析与年龄计算
    - 类别编码
    - 无用字段删除
    - 数值标准化
    - 根据卡号计算交易特征
    """

    # 1. 删除不适合建模的字段：如商户信息、街道地址、用户姓/名等
    columns_to_drop = ["merchant", "first", "last", "street", "city", "state", "zip", "trans_num", "unix_time", "city_pop", 'job']
    df = df.drop(columns=columns_to_drop, errors='ignore')  # 忽略不存在的列

    # 2. 将字符串时间转换为 datetime 类型
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors='coerce')
    df["dob"] = pd.to_datetime(df["dob"], errors='coerce')

    # 3. 从交易时间中提取出小时、天、月份等特征
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day
    #df["month"] = df["trans_date_trans_time"].dt.month

    # 4. 根据出生日期和交易日期计算年龄（单位为年）
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365

    # 5. 根据信用卡号统计每张卡的交易总次数
    total_transactions = df.groupby('cc_num')['trans_date_trans_time'].count().reset_index(name='total_transactions')
    df = df.merge(total_transactions, on='cc_num', how='left')  # 将交易次数合并回主数据集

    # 6. 计算过去30天内的平均交易频率（每天交易数）
    df['transaction_date'] = df['trans_date_trans_time'].dt.date
    recent_30_days = df.groupby('cc_num').apply(
        lambda x: x[x['transaction_date'] >= (x['transaction_date'].max() - pd.Timedelta(days=30))].shape[0] / 30
    ).reset_index(name='30_day_transaction_frequency')
    df = df.merge(recent_30_days, on='cc_num', how='left')

    # 7. 计算过去7天内的平均交易频率（每天交易数）
    recent_7_days = df.groupby('cc_num').apply(
        lambda x: x[x['transaction_date'] >= (x['transaction_date'].max() - pd.Timedelta(days=7))].shape[0] / 7
    ).reset_index(name='7_day_transaction_frequency')
    df = df.merge(recent_7_days, on='cc_num', how='left')


    # 8. 计算过去1天内的平均交易频率（每天交易数）
    recent_1_day = df.groupby('cc_num').apply(
        lambda x: x[x['transaction_date'] >= (x['transaction_date'].max() - pd.Timedelta(days=1))].shape[0] / 1
    ).reset_index(name='1_day_transaction_frequency')
    df = df.merge(recent_1_day, on='cc_num', how='left')

    # 9 计算距离上一次交易的时间（单位：小时）
    #df = df.sort_values(by=['cc_num', 'trans_date_trans_time'])  # 按卡号和时间排序
    #df['prev_trans_time'] = df.groupby('cc_num')['trans_date_trans_time'].shift(1)
    #df['hours_since_last_transaction'] = (df['trans_date_trans_time'] - df['prev_trans_time']).dt.total_seconds() / 3600
    #df['hours_since_last_transaction'] = df['hours_since_last_transaction'].fillna(-1)  # 第一次交易为 -1
    #df.drop(columns=['prev_trans_time'], inplace=True)  # 中间列删除


    # 8. 是否金额超过200美元：创建布尔型特征（1表示大于等于200）
    df['is_amt_greater_than_200'] = df['amt'].apply(lambda x: 1 if x >= 200 else 0)

    # 9. 将性别字段转为0/1，M 为 1，其他为 0
    df['gender'] = df['gender'].apply(M_1_F_0)

    # 10. 对 'category' 类别特征进行 One-Hot 编码（去除第一列以防止多重共线性）
    categorical_cols = ["category"]
    encoder = OneHotEncoder(drop='first', sparse_output=False)  # sparse_output=False 返回稠密数组

    for col in categorical_cols:
        encoded_array = encoder.fit_transform(df[[col]])
        encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out([col]))
        df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)  # 删除原列，拼接新列

    # 11. 根据交易时间的小时数映射时间段（夜间/白天），再进行编码
    df['time_category'] = df['hour'].apply(time_category)
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_array = encoder.fit_transform(df[['time_category']])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['time_category']))
    df = pd.concat([df.drop(columns=['time_category']), encoded_df], axis=1)

    # 12. 根据星期几映射为工作日/休息日，再进行编码
    df['category_day'] = df['day'].apply(category_day)
    encoded_array = encoder.fit_transform(df[['category_day']])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['category_day']))
    df = pd.concat([df.drop(columns=['category_day']), encoded_df], axis=1)

    # 13. 将编码后结果合并为一个最终的布尔值特征 category_day（是否为工作日）
    df['category_day'] = df['category_day_cat2']
    df.drop(columns=['category_day_cat2'], inplace=True)
    # 14. 分离特征与标签，去除标签列和不参与建模的辅助列
    if "is_fraud" in df.columns:
        y = df["is_fraud"]  # 标签列
        X = df.drop(columns=["is_fraud", "cc_num", "trans_date_trans_time", "dob", "transaction_date"])  # 特征集
    else:
        y = pd.Series([])
        X = df.drop(columns=["cc_num", "trans_date_trans_time", "dob", "transaction_date"])

    return X, y  # 返回处理后的特征和标签
