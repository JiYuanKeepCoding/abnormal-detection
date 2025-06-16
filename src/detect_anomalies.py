import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from compare_result import compare_results
from datetime import datetime
from holiday_utils import HolidayUtils

def create_time_features(df):
    """
    创建时间相关特征
    """
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['day_of_month'] = df['time'].dt.day
    df['month'] = df['time'].dt.month
    # 修改周末判断逻辑：周一早上0点也算周末
    df['is_weekend'] = ((df['day_of_week'].isin([5, 6])) | 
                       ((df['day_of_week'] == 0) & (df['hour'] == 0))).astype(int)
    df['is_month_start'] = df['time'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['time'].dt.is_month_end.astype(int)
    df['quarter'] = df['time'].dt.quarter
    df['is_workday'] = (~df['time'].dt.dayofweek.isin([5, 6])).astype(int)
    
    # 添加整天价格缺失的特征
    df['is_all_day_missing'] = df.groupby(['product_type1', 'product_type2', df['time'].dt.date])['price'].transform(
        lambda x: (x.notna().sum() == 0).astype(int)
    )
    
    # 计算时间特征
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    
    # 计算每个时间窗口的统计特征
    for window in [3, 7]:
        df[f'hour_ma_{window}'] = df.groupby(['product_type1', 'product_type2'])['hour'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'hour_std_{window}'] = df.groupby(['product_type1', 'product_type2'])['hour'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    
    # 计算与小时移动平均的偏差
    df['hour_ma_deviation'] = (df['hour'] - df['hour_ma_3']) / df['hour_std_3']
    
    return df

def detect_anomalies():
    """
    检测异常值
    """
    # 设置工作目录
    os.chdir('./abnormal-detection/data')
    
    # 读取数据
    df = pd.read_csv('complete_time_series.csv')
    
    # 创建时间特征
    df['time'] = pd.to_datetime(df['time'])
    df = create_time_features(df)

    df = df[df['price'].isna()]
    
    # 添加节假日特征
    df = HolidayUtils.add_holiday_features(df)
    
    # 定义特征
    time_features = [
        # 'day_of_week',
        'is_weekend', 
        'is_all_day_missing'
    ]
    
    # price_features = ['price']
    
    # 初始化标准化器
    scaler = StandardScaler()
    
    # 对每个产品组合分别训练模型
    all_predictions = []
    feature_importance = []
    
    for (prod1, prod2), group_data in df.groupby(['product_type1', 'product_type2']):
        if len(group_data) < 10:  # 跳过数据点太少的产品组合
            continue
            
        # 标准化价格特征
        # group_data[price_features] = scaler.fit_transform(group_data[price_features])
        
        # 合并所有特征
        feature_columns = time_features
        
        # 训练孤立森林模型
        model = IsolationForest(
            n_estimators=200,
            max_samples=min(256, len(group_data)),
            contamination=0.03,
            random_state=42,
            n_jobs=-1
        )
        
        # 预测异常
        predictions = model.fit_predict(group_data[feature_columns])
        
        # 将预测结果添加到数据中
        group_data['is_anomaly'] = predictions
        
        # 保存预测结果
        all_predictions.append(group_data)
    
    # 合并所有预测结果
    if all_predictions:
        final_df = pd.concat(all_predictions)
        
        # 只保存异常数据点
        anomalies_df = final_df[final_df['is_anomaly'] == -1].copy()
        
        # 保存结果
        anomalies_df.to_csv('detected_anomalies.csv', index=False)
        
        # 打印统计信息
        print("\n异常检测统计：")
        print(f"总数据点数量: {len(final_df)}")
        print(f"检测到的异常点数量: {len(anomalies_df)}")
        print(f"异常点比例: {len(anomalies_df) / len(final_df):.2%}")
        
        # 分析异常点的时间分布
        print("\n异常点的时间分布：")
        
        # 分析异常点在工作日/周末的分布
        print("\n工作日/周末分布：")
        print(anomalies_df['is_weekend'].value_counts(normalize=True))
        
        # 分析异常点在节假日的分布
        print("\n节假日分布：")
        print(anomalies_df['is_cn_holiday'].value_counts(normalize=True))
        
        # 可视化异常点分布
        plt.figure(figsize=(15, 10))
        
        # 工作日/周末分布
        plt.subplot(2, 2, 1)
        sns.countplot(data=anomalies_df, x='is_weekend')
        plt.title('Anomaly Distribution: Weekday vs Weekend')
        plt.xlabel('Is Weekend')
        plt.ylabel('Number of Anomalies')
        
        # 节假日分布
        plt.subplot(2, 2, 2)
        sns.countplot(data=anomalies_df, x='is_cn_holiday')
        plt.title('Anomaly Distribution: Holiday vs non Holiday')
        plt.xlabel('Is Holiday')
        plt.ylabel('Number of Anomalies')
        
        # 时间间隔分布
        plt.subplot(2, 2, 3)
        sns.histplot(data=anomalies_df, x='time_diff', bins=50)
        plt.title('Anomaly Distribution: Time Interval')
        plt.xlabel('Time Interval (seconds)')
        plt.ylabel('Number of Anomalies')
        
        # 保存图表
        plt.tight_layout()
        plt.savefig('../output/plots/anomaly_distribution.png')
        plt.close()
    else:
        print("没有足够的数据进行异常检测")

if __name__ == '__main__':
    detect_anomalies()
    
    # 读取检测结果和实际删除的数据进行比较
    detected_anomalies = pd.read_csv('detected_anomalies.csv')
    removed_data = pd.read_csv('removed.csv')
    
    # 调用比较函数
    print("\n开始比较检测结果和实际删除的数据...")
    compare_results(detected_anomalies, removed_data) 