import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_interval_mode(group):
    """
    计算时间间隔的众数
    
    Args:
        group: 按产品组合分组的数据
        
    Returns:
        时间间隔的众数（小时）
    """
    # 计算时间差（小时）
    time_diff = group['time'].diff().dt.total_seconds() / 3600
    # 返回众数
    return time_diff.mode()[0]

    """
    添加节假日特征
    
    Args:
        df: 数据框，包含 time 列
        
    Returns:
        添加了节假日特征的数据框
    """
    # 确保时间列为datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    
    # 添加日期列
    df['date'] = df['time'].dt.date.astype(str)
    
    # 添加节假日特征
    df['is_cn_holiday'] = df['date'].isin(HOLIDAYS['CN']).astype(int)
    df['is_us_holiday'] = df['date'].isin(HOLIDAYS['US']).astype(int)
    df['is_jp_holiday'] = df['date'].isin(HOLIDAYS['JP']).astype(int)
    
    # 删除临时日期列
    df = df.drop('date', axis=1)
    
    return df

def complete_time_series(df):
    """
    补全时间序列数据，对每个产品组合按照其时间间隔的众数来补全缺失的时间点
    
    Args:
        df: 原始数据框，包含 product_type1, product_type2, time, price, price2 列
        
    Returns:
        补全后的数据框，增加了 is_original 列标记原始数据
    """
    # 获取所有产品组合
    product_combinations = df[['product_type1', 'product_type2']].drop_duplicates()
    
    # 计算每个产品组合的时间间隔众数
    interval_modes = df.groupby(['product_type1', 'product_type2']).apply(get_interval_mode)
    
    # 创建完整的数据框
    complete_data = []
    
    for _, row in product_combinations.iterrows():
        prod1, prod2 = row['product_type1'], row['product_type2']
        
        # 获取该组合的现有数据
        existing_data = df[(df['product_type1'] == prod1) & (df['product_type2'] == prod2)].sort_values('time')
        
        if len(existing_data) < 2:
            # 如果数据点少于2个，无法计算间隔，直接添加原始数据
            for _, data_row in existing_data.iterrows():
                complete_data.append({
                    'product_type1': prod1,
                    'product_type2': prod2,
                    'time': data_row['time'],
                    'price': data_row['price'],
                    'price2': data_row['price2'],
                    'is_original': True
                })
            continue
            
        # 获取该组合的时间间隔众数
        interval = interval_modes[(prod1, prod2)]
        
        # 获取时间范围
        start_time = existing_data['time'].min()
        end_time = existing_data['time'].max()
        
        # 创建该组合的完整时间序列
        current_time = start_time
        while current_time <= end_time:
            # 检查这个时间点是否已有数据
            existing_row = existing_data[existing_data['time'] == current_time]
            
            if len(existing_row) > 0:
                # 如果已有数据，保留原有数据
                complete_data.append({
                    'product_type1': prod1,
                    'product_type2': prod2,
                    'time': current_time,
                    'price': existing_row['price'].iloc[0],
                    'is_original': True
                })
            else:
                # 如果没有数据，添加空值
                complete_data.append({
                    'product_type1': prod1,
                    'product_type2': prod2,
                    'time': current_time,
                    'price': None,
                    'is_original': False
                })
            
            # 增加时间间隔
            current_time += pd.Timedelta(hours=interval)
    
    # 创建完整的数据框
    complete_df = pd.DataFrame(complete_data)
    
    def calculate_time_diff(group):
        # 对每个时间点，找到最近的前一个有价格数据的时间点
        time_diffs = []
        last_price_time = None
        for idx, row in group.iterrows():
            # if row['product_type1'] == 'A' and row['product_type2'] == 'B' and str(row['time']) == '2024-03-26 18:00:00':
            #     print("xxx")
            if last_price_time is not None:
                # 计算与最近的前一个有价格数据的时间差（小时）
                time_diff = (row['time'] - last_price_time).total_seconds() / 3600
                time_diffs.append(time_diff)
            else:
                time_diffs.append(None)  # 之前没有价格数据
            
            if pd.notna(row['price']):  # 使用pd.notna()检查NaN
                last_price_time = row['time']
                
        return pd.Series(time_diffs, index=group.index)
    
    # 按产品组合和时间排序
    complete_df = complete_df.sort_values(['product_type1', 'product_type2', 'time'])
    
    # 计算time_diff
    time_diffs = []
    for (prod1, prod2), group in complete_df.groupby(['product_type1', 'product_type2']):
        # 计算这个产品组合的时间差
        diffs = calculate_time_diff(group)
        # 把每个时间差添加到列表中
        time_diffs.extend(diffs.values)
    
    # 确保time_diffs的长度与complete_df相同
    assert len(time_diffs) == len(complete_df), "时间差列表长度与数据框长度不匹配"
    
    # 添加time_diff列
    complete_df['time_diff'] = time_diffs
    
    return complete_df

if __name__ == '__main__':
    # 读取原始数据
    os.getcwd()
    os.chdir('abnormal-detection/data')
    df = pd.read_csv('mock_data.csv')
    df['time'] = pd.to_datetime(df['time'])
    
    # 补全时间序列
    complete_df = complete_time_series(df)
    
    # 保存补全后的数据
    complete_df.to_csv('complete_time_series.csv', index=False)
    print("已生成补全的时间序列数据，保存到 complete_time_series.csv")
    
    # 打印统计信息
    print(f"\n总数据点数量: {len(complete_df)}")
    print(f"原始数据点数量: {len(complete_df[complete_df['is_original']])}")
    print(f"补全的数据点数量: {len(complete_df[~complete_df['is_original']])}")
 
    # 打印每个产品组合的时间间隔
    print("\n各产品组合的时间间隔（小时）：")
    interval_modes = df.groupby(['product_type1', 'product_type2']).apply(get_interval_mode)
    for (prod1, prod2), interval in interval_modes.items():
        print(f"{prod1}-{prod2}: {interval:.2f}小时") 