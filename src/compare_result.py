import pandas as pd
import os
from datetime import datetime

def create_key(row):
    """
    为数据行创建唯一标识符
    
    Args:
        row: 数据行，包含 product_type1, product_type2, time 列
        
    Returns:
        str: 唯一标识符
    """
    return f"{row['product_type1']}_{row['product_type2']}_{row['time']}"

def calculate_metrics(tp, fp, fn):
    """
    计算评估指标
    
    Args:
        tp: 真正例数量
        fp: 假正例数量
        fn: 假负例数量
        
    Returns:
        dict: 包含precision, recall, f1_score的字典
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def compare_results(detected_anomalies, removed_data):
    """
    比较检测到的异常和实际删除的数据
    
    Args:
        detected_anomalies: 检测到的异常数据框
        removed_data: 实际删除的数据框
    """
    # 确保时间格式一致
    detected_anomalies['time'] = pd.to_datetime(detected_anomalies['time'])
    removed_data['time'] = pd.to_datetime(removed_data['time'])
    
    # 创建唯一标识符集合
    detected_keys = set(detected_anomalies.apply(create_key, axis=1))
    removed_keys = set(removed_data.apply(create_key, axis=1))
    
    # 计算评估指标
    true_positives = len(detected_keys & removed_keys)
    false_positives = len(detected_keys - removed_keys)
    false_negatives = len(removed_keys - detected_keys)
    
    # 计算整体评估指标
    metrics = calculate_metrics(true_positives, false_positives, false_negatives)
    
    # 打印整体评估结果
    print("\n异常检测评估结果：")
    print(f"真正例（正确检测的异常）: {true_positives}")
    print(f"假正例（误报）: {false_positives}")
    print(f"假负例（漏报）: {false_negatives}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1_score']:.4f}")
    
    # 找出漏报数据（实际删除但未检测到）
    false_negatives_data = removed_data[removed_data.apply(create_key, axis=1).isin(removed_keys - detected_keys)]
    false_negatives_data.to_csv('false_negatives.csv', index=False)
    
    # 找出误报数据（检测到但实际未删除）
    false_positives_data = detected_anomalies[detected_anomalies.apply(create_key, axis=1).isin(detected_keys - removed_keys)]
    false_positives_data.to_csv('false_positives.csv', index=False)
    
    # 按产品组合分析
    print("\n各产品组合的评估结果：")
    product_combinations = set(zip(detected_anomalies['product_type1'], detected_anomalies['product_type2']))
    
    for prod1, prod2 in product_combinations:
        # 获取该组合的检测结果和实际数据
        detected_group = detected_anomalies[
            (detected_anomalies['product_type1'] == prod1) & 
            (detected_anomalies['product_type2'] == prod2)
        ]
        removed_group = removed_data[
            (removed_data['product_type1'] == prod1) & 
            (removed_data['product_type2'] == prod2)
        ]
        
        # 创建该组合的唯一标识符集合
        detected_keys = set(detected_group.apply(create_key, axis=1))
        removed_keys = set(removed_group.apply(create_key, axis=1))
        
        # 计算该组合的评估指标
        tp = len(detected_keys & removed_keys)
        fp = len(detected_keys - removed_keys)
        fn = len(removed_keys - detected_keys)
        
        metrics = calculate_metrics(tp, fp, fn)
        
        print(f"\n产品组合 {prod1}-{prod2}:")
        print(f"真正例: {tp}")
        print(f"假正例: {fp}")
        print(f"假负例: {fn}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1_score']:.4f}")
    
    # 保存评估结果
    results = {
        'overall': {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            **metrics
        }
    }
    
    # 保存结果到CSV
    results_df = pd.DataFrame([{
        'product_combination': 'overall',
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score']
    }])
    
    for prod1, prod2 in product_combinations:
        detected_group = detected_anomalies[
            (detected_anomalies['product_type1'] == prod1) & 
            (detected_anomalies['product_type2'] == prod2)
        ]
        removed_group = removed_data[
            (removed_data['product_type1'] == prod1) & 
            (removed_data['product_type2'] == prod2)
        ]
        
        detected_keys = set(detected_group.apply(create_key, axis=1))
        removed_keys = set(removed_group.apply(create_key, axis=1))
        
        tp = len(detected_keys & removed_keys)
        fp = len(detected_keys - removed_keys)
        fn = len(removed_keys - detected_keys)
        
        metrics = calculate_metrics(tp, fp, fn)
        
        results_df = pd.concat([results_df, pd.DataFrame([{
            'product_combination': f"{prod1}-{prod2}",
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        }])], ignore_index=True)
    
    # 保存到CSV文件
    results_df.to_csv('evaluation_metrics.csv', index=False)
    print("\n评估结果已保存到 evaluation_metrics.csv")
    print("漏报数据已保存到 false_negatives.csv")
    print("误报数据已保存到 false_positives.csv")

if __name__ == '__main__':
    # 读取检测结果和实际删除的数据
    os.chdir('./abnormal-detection/data')
    detected_anomalies = pd.read_csv('detected_anomalies.csv')
    removed_data = pd.read_csv('removed.csv')
    
    # 比较结果
    compare_results(detected_anomalies, removed_data) 