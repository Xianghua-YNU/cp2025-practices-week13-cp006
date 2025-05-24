import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

def load_data(filepath):
    """从文本文件加载一维数据，每行一个数"""
    with open(filepath, 'r') as f:
        data = [float(line.strip()) for line in f if line.strip()]
    return np.array(data)

def generate_dates(num_days):
    """生成从2006年7月1日开始的日期序列"""
    start_date = datetime(2006, 7, 1)
    return [start_date + timedelta(days=i) for i in range(num_days)]

def plot_series(dates, data, title, filename, label='Original Data', filtered=None, filtered_label=None):
    """绘制时间序列图，可选叠加滤波结果"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, data, 'b-', alpha=0.7, label=label)
    if filtered is not None:
        ax.plot(dates, filtered, 'r-', label=filtered_label)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Index Value', fontsize=12)
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Image saved: {filename}")
    plt.close(fig)

def fourier_filter(data, retention_ratio):
    """保留前retention_ratio比例的低频系数"""
    coeff = np.fft.rfft(data)
    cutoff = int(len(coeff) * retention_ratio)
    coeff[cutoff:] = 0
    filtered = np.fft.irfft(coeff, n=len(data))
    return filtered

def main():
    # 路径设置
    data_path = r'C:\Users\31025\OneDrive\桌面\t\dow.txt'
    save_dir = r'C:\Users\31025\OneDrive\桌面\t'
    os.makedirs(save_dir, exist_ok=True)

    # 1. 数据加载
    data = load_data(data_path)
    dates = generate_dates(len(data))

    # 2. 绘制原始数据
    plot_series(
        dates, data,
        title='Dow Jones Index Original Time Series',
        filename=os.path.join(save_dir, 'dow_original.png')
    )

    # 3. 傅立叶滤波（保留前10%系数）
    filtered_10 = fourier_filter(data, 0.1)
    plot_series(
        dates, data,
        title='Fourier Filtered (Retain 10%) vs Original',
        filename=os.path.join(save_dir, 'dow_filter_10.png'),
        filtered=filtered_10,
        filtered_label='Filtered (10% Low Frequency)'
    )

    # 4. 傅立叶滤波（保留前2%系数）
    filtered_2 = fourier_filter(data, 0.02)
    plot_series(
        dates, data,
        title='Fourier Filtered (Retain 2%) vs Original',
        filename=os.path.join(save_dir, 'dow_filter_2.png'),
        filtered=filtered_2,
        filtered_label='Filtered (2% Low Frequency)'
    )

if __name__ == "__main__":
    main()
