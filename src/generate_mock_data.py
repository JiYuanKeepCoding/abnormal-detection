import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
from holiday_utils import HolidayUtils

def generate_mock_data():
    """
    生成模拟数据
    """
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 设置工作目录
    os.chdir('./abnormal-detection/data')
    
    # 生成产品类型
    product_types = ['A', 'B', 'C', 'D']
    
    # 生成时间序列（2024年全年，每小时一个数据点）
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31, 23, 0)
    
    # 生成所有产品组合
    product_combinations = []
    for prod1 in product_types:
        for prod2 in product_types:
            if prod1 != prod2:
                product_combinations.append((prod1, prod2))
    
    # 为每个产品组合生成固定的时间间隔（1-10
    interval_dict = {combo: random.randint(1, 10) for combo in product_combinations}
    
    # 生成数据
    data = []
    removed_data = []
    
    for prod1, prod2 in product_combinations:
        # 为每个产品组合生成基础价格
        base_price = random.uniform(100, 1000)
        
        # 获取这个产品组合的时间间隔
        interval = interval_dict[(prod1, prod2)]
        
        # 生成时间序列
        current_time = start_date
        while current_time <= end_date:
            # 检查当前日期是否是节假日或周末
            current_date_str = current_time.strftime('%Y-%m-%d')
            is_holiday = any([
                HolidayUtils.is_holiday(current_date_str, 'CN'),
                HolidayUtils.is_holiday(current_date_str, 'US'),
                HolidayUtils.is_holiday(current_date_str, 'JP')
            ])
            is_weekend = current_time.weekday() >= 5  # 5和6分别代表周六和周日
            
            # 如果不是节假日和周末，则生成数据
            if not (is_holiday or is_weekend):
                # 生成随机波动
                price = base_price * (1 + random.uniform(-0.1, 0.1))
                
                # 随机决定是否删除这个数据点（约1%的概率）
                if random.random() < 0.01:
                    removed_data.append({
                        'product_type1': prod1,
                        'product_type2': prod2,
                        'time': current_time,
                        'price': price
                    })
                else:
                    data.append({
                        'product_type1': prod1,
                        'product_type2': prod2,
                        'time': current_time,
                        'price': price
                    })
            
            # 移动到下一个时间点（使用产品组合的固定间隔）
            current_time += timedelta(hours=interval)
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    removed_df = pd.DataFrame(removed_data)
    
    # 保存数据
    df.to_csv('mock_data.csv', index=False)
    removed_df.to_csv('removed.csv', index=False)
    
    # 打印统计信息
    print("\n数据生成统计：")
    print(f"总数据点数量: {len(df)}")
    print(f"删除的数据点数量: {len(removed_df)}")
    print(f"产品组合数量: {len(product_combinations)}")
    
    # 打印每个产品组合的时间间隔
    print("\n各产品组合的时间间隔：")
    for (prod1, prod2), interval in interval_dict.items():
        print(f"{prod1}-{prod2}: {interval}小时")
    
    # 分析删除的数据点分布
    if not removed_df.empty:
        print("\n删除的数据点分布：")
        print(removed_df.groupby(['product_type1', 'product_type2']).size())

if __name__ == '__main__':
    generate_mock_data() 