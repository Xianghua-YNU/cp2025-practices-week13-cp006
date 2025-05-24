import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft

def load_data(file_path):
    """加载数据文件，返回时间和信号数组"""
    data = np.loadtxt(file_path)
    # 处理一维数据情况
    if data.ndim == 1:
        time = np.arange(len(data))
        signal = data
    else:
        time = data[:, 0]
        signal = data[:, 1]
    return time, signal

def plot_data(time, signal, title="原始信号"):
    """绘制原始信号"""
    # 支持只传入signal的情况
    if isinstance(time, np.ndarray) and isinstance(signal, str):
        title = signal
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
    """
    应用傅里叶滤波，保留指定比例的最大系数
    
    参数:
        signal: 信号数组
