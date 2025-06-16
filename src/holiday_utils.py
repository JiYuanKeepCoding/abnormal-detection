import pandas as pd
from datetime import datetime, timedelta

class HolidayUtils:
    """节假日工具类，用于处理节假日相关的功能"""
    
    # 2024年节假日定义
    HOLIDAYS = {
        'CN': [
            '2024-01-01',  # 元旦
            '2024-02-10', '2024-02-11', '2024-02-12', '2024-02-13', '2024-02-14', '2024-02-15', '2024-02-16', '2024-02-17',  # 春节
            '2024-04-04', '2024-04-05', '2024-04-06',  # 清明节
            '2024-05-01', '2024-05-02', '2024-05-03', '2024-05-04', '2024-05-05',  # 劳动节
            '2024-06-10',  # 端午节
            '2024-09-15', '2024-09-16', '2024-09-17',  # 中秋节
            '2024-10-01', '2024-10-02', '2024-10-03', '2024-10-04', '2024-10-05', '2024-10-06', '2024-10-07'  # 国庆节
        ],
        'US': [
            '2024-01-01',  # New Year's Day
            '2024-01-15',  # Martin Luther King Jr. Day
            '2024-02-19',  # Presidents' Day
            '2024-05-27',  # Memorial Day
            '2024-06-19',  # Juneteenth
            '2024-07-04',  # Independence Day
            '2024-09-02',  # Labor Day
            '2024-10-14',  # Columbus Day
            '2024-11-11',  # Veterans Day
            '2024-11-28',  # Thanksgiving Day
            '2024-12-25'   # Christmas Day
        ],
        'JP': [
            '2024-01-01',  # 元日
            '2024-01-08',  # 成人の日
            '2024-02-12',  # 建国記念の日
            '2024-02-23',  # 天皇誕生日
            '2024-03-20',  # 春分の日
            '2024-04-29',  # 昭和の日
            '2024-05-03', '2024-05-04', '2024-05-05', '2024-05-06',  # ゴールデンウィーク
            '2024-07-15',  # 海の日
            '2024-08-11',  # 山の日
            '2024-08-12',  # 振替休日
            '2024-09-16',  # 敬老の日
            '2024-09-22',  # 秋分の日
            '2024-09-23',  # 振替休日
            '2024-11-03',  # 文化の日
            '2024-11-04',  # 振替休日
            '2024-11-23'   # 勤労感謝の日
        ]
    }
    
    @classmethod
    def calculate_holiday_weekend_ratio(cls, year=2024):
        """
        计算指定年份的节假日和周末占比
        
        Args:
            year: 年份，默认为2024
            
        Returns:
            dict: 包含各种统计信息的字典
        """
        # 计算总天数
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        total_days = (end_date - start_date).days + 1
        
        # 计算周末天数
        weekend_days = 0
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() >= 5:  # 5和6分别代表周六和周日
                weekend_days += 1
            current_date += timedelta(days=1)
        
        # 计算节假日天数
        holiday_days = len(cls.HOLIDAYS['CN'])
        
        # 计算节假日和周末重叠的天数
        overlap_days = 0
        for holiday in cls.HOLIDAYS['CN']:
            holiday_date = datetime.strptime(holiday, '%Y-%m-%d')
            if holiday_date.weekday() >= 5:
                overlap_days += 1
        
        # 计算实际休息日总数（周末 + 节假日 - 重叠）
        total_rest_days = weekend_days + holiday_days - overlap_days
        
        return {
            'total_days': total_days,
            'weekend_days': weekend_days,
            'holiday_days': holiday_days,
            'overlap_days': overlap_days,
            'total_rest_days': total_rest_days,
            'weekend_ratio': weekend_days / total_days * 100,
            'holiday_ratio': holiday_days / total_days * 100,
            'total_rest_ratio': total_rest_days / total_days * 100
        }
    
    @classmethod
    def add_holiday_features(cls, df):
        """
        为数据框添加节假日特征
        
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
        df['is_cn_holiday'] = df['date'].isin(cls.HOLIDAYS['CN']).astype(int)
        df['is_us_holiday'] = df['date'].isin(cls.HOLIDAYS['US']).astype(int)
        df['is_jp_holiday'] = df['date'].isin(cls.HOLIDAYS['JP']).astype(int)
        
        # 删除临时日期列
        df = df.drop('date', axis=1)
        
        return df
    
    @classmethod
    def is_holiday(cls, date, country):
        """
        判断指定日期是否是指定国家的节假日
        
        Args:
            date: 日期，可以是字符串或datetime对象
            country: 国家代码，'CN'、'US'或'JP'
            
        Returns:
            bool: 是否为节假日
        """
        if isinstance(date, datetime):
            date = date.strftime('%Y-%m-%d')
        return date in cls.HOLIDAYS.get(country, [])
    
    @classmethod
    def get_holidays(cls, country):
        """
        获取指定国家的节假日列表
        
        Args:
            country: 国家代码，'CN'、'US'或'JP'
            
        Returns:
            list: 节假日日期列表
        """
        return cls.HOLIDAYS.get(country, [])
    
    @classmethod
    def get_holiday_stats(cls, df):
        """
        获取数据框中的节假日统计信息
        
        Args:
            df: 数据框，包含节假日特征列
            
        Returns:
            dict: 节假日统计信息
        """
        return {
            'cn_holiday_count': df['is_cn_holiday'].sum(),
            'us_holiday_count': df['is_us_holiday'].sum(),
            'jp_holiday_count': df['is_jp_holiday'].sum()
        }

if __name__ == '__main__':
    # 计算2024年的节假日和周末占比
    stats = HolidayUtils.calculate_holiday_weekend_ratio(2024)
    print("\n2024年节假日和周末统计：")
    print(f"总天数: {stats['total_days']}天")
    print(f"周末天数: {stats['weekend_days']}天 ({stats['weekend_ratio']:.2f}%)")
    print(f"节假日天数: {stats['holiday_days']}天 ({stats['holiday_ratio']:.2f}%)")
    print(f"节假日和周末重叠天数: {stats['overlap_days']}天")
    print(f"实际休息日总数: {stats['total_rest_days']}天 ({stats['total_rest_ratio']:.2f}%)") 