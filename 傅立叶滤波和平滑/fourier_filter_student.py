
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft

def load_data(file_path):
    """加载数据文件，返回时间和信号数组"""
    data = np.loadtxt(file_path)
    time = data[:, 0]
    signal = data[:, 1]
    return time, signal

def plot_data(time, signal, title="原始信号"):
    """绘制原始信号"""
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel('时间')
    plt.ylabel('幅度')
    plt.grid(True)
    return plt.gcf()

def fourier_filter(time, signal, cutoff_freq):
    """
    应用傅里叶滤波，去除高于截止频率的成分
    
    参数:
        time: 时间数组
        signal: 信号数组
        cutoff_freq: 截止频率
        
    返回:
        滤波后的信号
    """
    # 计算采样间隔
    dt = time[1] - time[0]
    n = len(signal)
    
    # 执行傅里叶变换
    yf = fft(signal)
    xf = fftfreq(n, dt)
    
    # 创建滤波器
    mask = np.abs(xf) <= cutoff_freq
    
    # 应用滤波器
    yf_filtered = yf * mask
    
    # 执行逆变换
    filtered_signal = ifft(yf_filtered).real
    
    return filtered_signal

def plot_comparison(time, original, filtered, title="滤波前后对比"):
    """对比绘制原始信号和滤波后的信号"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(time, original)
    plt.title('原始信号')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time, filtered)
    plt.title('滤波后信号')
    plt.xlabel('时间')
    plt.grid(True)
    
    plt.tight_layout()
    return plt.gcf()    
