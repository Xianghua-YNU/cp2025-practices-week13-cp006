import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

def load_data(file_path):
    """加载数据文件，返回时间和信号数组
    
    参数:
        file_path (str): 数据文件路径
        
    返回:
        tuple: (time, signal)，其中time为时间数组，signal为信号数组
    """
    try:
        data = np.loadtxt(file_path)
        if data.ndim == 1:  # 处理只有信号值的情况
            time = np.arange(len(data))
            signal = data
        else:
            time = data[:, 0]
            signal = data[:, 1]
        return time, signal
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def plot_data(time, signal=None, title="原始信号"):
    """绘制原始信号
    
    支持两种调用方式：
    1. plot_data(time, signal) - 传入时间和信号
    2. plot_data(signal) - 此时time默认为索引，title为默认值
    
    参数:
        time (array-like): 时间数组（可选，当signal为信号数组时传入）
        signal (array-like): 信号数组
        title (str): 图表标题（默认值："原始信号"）
        
    返回:
        matplotlib.figure.Figure: 绘制的图形对象
    """
    if signal is None:
        signal = time
        time = np.arange(len(signal))
        
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel('时间')
    plt.ylabel('幅度')
    plt.grid(True)
    return plt.gcf()

def fourier_filter(signal, threshold):
    """应用傅里叶滤波，保留指定比例的最大系数
    
    参数:
        signal (array-like): 输入信号数组
        threshold (float): 保留系数的比例（0.0-1.0，例如0.1表示保留前10%的系数）
        
    返回:
        tuple: (filtered_signal, fourier_coefficients)，其中filtered_signal为滤波后的信号，fourier_coefficients为傅里叶系数
    """
    n = len(signal)
    if n == 0:
        raise ValueError("输入信号为空")
    
    # 执行傅里叶变换
    yf = fft(signal)
    
    # 计算系数幅度并排序
    amplitudes = np.abs(yf)
    sorted_indices = np.argsort(amplitudes)[::-1]  # 降序排列
    
    # 确定保留的系数数量
    num_keep = max(1, int(n * threshold))  # 至少保留1个系数
    mask = np.zeros(n, dtype=bool)
    mask[sorted_indices[:num_keep]] = True
    
    # 应用掩码并执行逆变换
    yf_filtered = yf * mask
    filtered_signal = ifft(yf_filtered).real
    
    return filtered_signal, yf

def plot_comparison(original_signal, filtered_signal, title="滤波前后对比"):
    """对比绘制原始信号和滤波后的信号
    
    参数:
        original_signal (array-like): 原始信号数组
        filtered_signal (array-like): 滤波后的信号数组
        title (str): 图表标题（默认值："滤波前后对比"）
        
    返回:
        matplotlib.figure.Figure: 绘制的图形对象
    """
    time = np.arange(len(original_signal))
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(time, original_signal)
    plt.title('原始信号')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time, filtered_signal)
    plt.title('滤波后信号')
    plt.xlabel('时间')
    plt.grid(True)
    
    plt.tight_layout()
    return plt.gcf()
