import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from 实习内容 import magnification_factor_and_coriolis_parameter

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def visualize_parameters():
    # 创建网格点
    x = np.linspace(0, 100, 50)  # 0-100范围内的50个点
    y = np.linspace(0, 100, 50)  # 0-100范围内的50个点
    X, Y = np.meshgrid(x, y)
    
    # 计算每个网格点的放大系数和科里奥利参数
    M = np.zeros_like(X)
    F = np.zeros_like(X)
    
    # 计算经纬度（假设标准纬度70°N，中心经度为100°E）
    standard_lat = 70
    center_lon = 100
    lat = standard_lat - Y * 0.1  # 每个网格点对应约0.1度纬度变化
    lon = center_lon + X * 0.1  # 每个网格点对应约0.1度经度变化
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            m, f = magnification_factor_and_coriolis_parameter('stereographic', X[i,j], Y[i,j], 100)
            M[i,j] = m
            F[i,j] = f
    
    # 创建图形 - 三维分布
    fig1 = plt.figure(figsize=(15, 6))
    
    # 绘制放大系数的3D曲面图
    ax1 = fig1.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(lon, lat, M, cmap='viridis')
    ax1.set_xlabel('经度 (°E)')
    ax1.set_ylabel('纬度 (°N)')
    ax1.set_zlabel('放大系数 (m)')
    ax1.set_title('放大系数的三维分布')
    fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # 绘制科里奥利参数的3D曲面图
    ax2 = fig1.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(lon, lat, F, cmap='viridis')
    ax2.set_xlabel('经度 (°E)')
    ax2.set_ylabel('纬度 (°N)')
    ax2.set_zlabel('科里奥利参数 (f)')
    ax2.set_title('科里奥利参数的三维分布')
    fig1.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    
    # 创建图形 - 等值线分布
    fig2 = plt.figure(figsize=(15, 6))
    
    # 绘制放大系数的等值线图
    ax3 = fig2.add_subplot(121)
    contour1 = ax3.contourf(lon, lat, M, levels=20, cmap='viridis')
    ax3.set_xlabel('经度 (°E)')
    ax3.set_ylabel('纬度 (°N)')
    ax3.set_title('放大系数的等值线分布')
    fig2.colorbar(contour1, ax=ax3)
    
    # 绘制科里奥利参数的等值线图
    ax4 = fig2.add_subplot(122)
    contour2 = ax4.contourf(lon, lat, F, levels=20, cmap='viridis')
    ax4.set_xlabel('经度 (°E)')
    ax4.set_ylabel('纬度 (°N)')
    ax4.set_title('科里奥利参数的等值线分布')
    fig2.colorbar(contour2, ax=ax4)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("开始可视化分析放大系数和科里奥利参数...")
    visualize_parameters()