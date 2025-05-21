#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
傅立叶滤波和平滑 - 道琼斯工业平均指数分析解决方案

本模块实现了对道琼斯工业平均指数数据的傅立叶分析和滤波处理。
"""

import numpy as np  # 导入数值计算库
import matplotlib.pyplot as plt  # 导入绘图库

def load_data(filename):
    """
    加载道琼斯工业平均指数数据
    
    参数:
        filename (str): 数据文件路径
    
    返回:
        numpy.ndarray: 指数数组
    """
    try:
        return np.loadtxt(filename)  # 从文件加载数据并返回
    except FileNotFoundError:  # 捕获文件不存在异常
        print(f"错误: 文件 '{filename}' 未找到，请确保文件路径正确。")
        raise  # 重新抛出异常，终止程序
    except Exception as e:  # 捕获其他异常
        print(f"加载数据时出错: {str(e)}")
        raise  # 重新抛出异常

def plot_data(data, title="Dow Jones Industrial Average"):
    """
    绘制时间序列数据
    """
    fig = plt.figure(figsize=(12, 6))  # 创建图形对象，设置尺寸
    plt.plot(data, linewidth=1.5, color='#1f77b4')  # 绘制数据线，设置线宽和颜色
    plt.title(title, fontsize=14)  # 设置标题和字体大小
    plt.xlabel("Trading Day", fontsize=12)  # 设置x轴标签和字体大小
    plt.ylabel("Index Value", fontsize=12)  # 设置y轴标签和字体大小
    plt.grid(True, alpha=0.3, linestyle='--')  # 添加网格线，设置透明度和线型
    plt.tight_layout()  # 自动调整布局
    plt.show()  # 显示图形
    return fig  # 返回图形对象

def fourier_filter(data, keep_fraction=0.1):
    """
    执行傅立叶变换并滤波
    
    参数:
        data (numpy.ndarray): 输入数据数组
        keep_fraction (float): 保留的傅立叶系数比例
    
    返回:
        tuple: (滤波后的数据数组, 原始傅立叶系数数组)
    """
    # 计算实数信号的傅立叶变换（只计算正频率部分）
    fft_coeff = np.fft.rfft(data)
    
    # 计算保留的系数数量（基于保留比例）
    cutoff = int(len(fft_coeff) * keep_fraction)
    
    # 创建滤波后的系数数组，复制原始系数
    filtered_coeff = fft_coeff.copy()
    filtered_coeff[cutoff:] = 0  # 将截止频率后的系数置为0（低通滤波）
    
    # 执行逆傅立叶变换，恢复时域信号
    filtered_data = np.fft.irfft(filtered_coeff, n=len(data))
    
    return filtered_data, fft_coeff  # 返回滤波后数据和原始傅立叶系数

def plot_comparison(original, filtered, title="Fourier Filter Result", keep_percentage=None):
    """
    绘制原始数据和滤波结果的比较
    """
    fig = plt.figure(figsize=(12, 6))  # 创建图形对象
    # 绘制原始数据，绿色线，线宽1，透明度0.7
    plt.plot(original, 'g-', linewidth=1, alpha=0.7, label="Original Data")
    # 绘制滤波后数据，红色线，线宽2
    plt.plot(filtered, 'r-', linewidth=2, label=f"Filtered ({keep_percentage}% coefficients)")
    plt.title(title, fontsize=14)  # 设置标题
    plt.xlabel("Trading Day", fontsize=12)  # 设置x轴标签
    plt.ylabel("Index Value", fontsize=12)  # 设置y轴标签
    plt.legend(fontsize=10)  # 显示图例
    plt.grid(True, alpha=0.3, linestyle='--')  # 添加网格线
    plt.tight_layout()  # 调整布局
    plt.show()  # 显示图形
    return fig  # 返回图形对象

def plot_spectrum(fft_coeff, title="Frequency Spectrum"):
    """
    绘制傅立叶系数的频谱图
    """
    fig = plt.figure(figsize=(12, 6))  # 创建图形对象
    # 计算频率值（对应每个傅立叶系数）
    frequencies = np.fft.rfftfreq(len(fft_coeff) * 2 - 2, d=1)
    magnitudes = np.abs(fft_coeff)  # 计算系数的幅度（绝对值）
    
    plt.plot(frequencies, magnitudes, 'b-', linewidth=1.5)  # 绘制频谱图
    plt.title(title, fontsize=14)  # 设置标题
    plt.xlabel("Frequency (cycles per day)", fontsize=12)  # 设置x轴标签
    plt.ylabel("Magnitude", fontsize=12)  # 设置y轴标签
    plt.grid(True, alpha=0.3, linestyle='--')  # 添加网格线
    plt.tight_layout()  # 调整布局
    plt.show()  # 显示图形
    return fig  # 返回图形对象

def calculate_rmse(original, filtered):
    """
    计算原始数据和滤波数据之间的均方根误差
    """
    return np.sqrt(np.mean((original - filtered) ** 2))  # 计算RMSE

def main():
    try:
        # 任务1：数据加载与可视化
        data = load_data('dow.txt')  # 加载数据
        print(f"数据加载成功，共{len(data)}个交易日数据")  # 打印数据信息
        plot_data(data, "Dow Jones Industrial Average - Original Data")  # 绘制原始数据图
        
        # 任务2：傅立叶变换与滤波（保留前10%系数）
        filtered_10, coeff = fourier_filter(data, 0.1)  # 执行滤波（保留10%系数）
        rmse_10 = calculate_rmse(data, filtered_10)  # 计算RMSE
        plot_comparison(data, filtered_10, "Fourier Filter Comparison", 10)  # 绘制对比图
        plot_spectrum(coeff, "Frequency Spectrum of Original Data")  # 绘制频谱图
        print(f"保留10%系数的RMSE: {rmse_10:.2f}")  # 打印RMSE值
        
        # 任务3：修改滤波参数（保留前2%系数）
        filtered_2, _ = fourier_filter(data, 0.02)  # 执行滤波（保留2%系数）
        rmse_2 = calculate_rmse(data, filtered_2)  # 计算RMSE
        plot_comparison(data, filtered_2, "Fourier Filter Comparison", 2)  # 绘制对比图
        print(f"保留2%系数的RMSE: {rmse_2:.2f}")  # 打印RMSE值
        
        # 任务4：自定义滤波参数（保留前50%系数）
        filtered_50, _ = fourier_filter(data, 0.5)  # 执行滤波（保留50%系数）
        rmse_50 = calculate_rmse(data, filtered_50)  # 计算RMSE
        plot_comparison(data, filtered_50, "Fourier Filter Comparison", 50)  # 绘制对比图
        print(f"保留50%系数的RMSE: {rmse_50:.2f}")  # 打印RMSE值
        
        # 任务5：对数变换后滤波
        log_data = np.log(data)  # 对数据取自然对数
        filtered_log, _ = fourier_filter(log_data, 0.1)  # 对对数数据滤波
        plot_comparison(log_data, filtered_log, "Fourier Filter on Log-transformed Data", 10)  # 绘制对比图
        
    except Exception as e:  # 捕获所有异常
        print(f"程序执行出错: {str(e)}")  # 打印错误信息

if __name__ == "__main__":
    main()  # 程序入口点
