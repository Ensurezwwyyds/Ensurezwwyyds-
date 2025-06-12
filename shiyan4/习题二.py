import numpy as np
import matplotlib.pyplot as plt
from 实习内容 import magnification_factor_and_coriolis_parameter

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def calculate_grid_parameters():
    # 设置网格参数
    d = 100  # 网格间距（km）
    rows = 6  # 向北的网格点数
    cols = 7  # 向东的网格点数
    proj = 'stereographic'
    
    # 创建网格点坐标矩阵
    lat_start = 50  # 起点纬度（°N）
    lon_start = 100  # 起点经度（°E）
    
    # 存储计算结果
    m_values = np.zeros((rows, cols))
    f_values = np.zeros((rows, cols))
    
    # 为每个网格点计算参数
    for i in range(rows):
        for j in range(cols):
            # 计算当前网格点的坐标
            lat = lat_start + i
            lon = lon_start + j
            
            # 计算放大系数和科里奥利参数
            m, f = magnification_factor_and_coriolis_parameter(proj, lat, lon, d)
            m_values[i, j] = m
            f_values[i, j] = f
            
            print(f"网格点 ({lon}°E, {lat}°N) 的计算结果：")
            print(f"放大系数 (m): {m:.6f}")
            print(f"科里奥利参数 (f): {f:.6e}\n")
    
    # 创建网格点的经纬度坐标
    lons = np.array([lon_start + j for j in range(cols)])
    lats = np.array([lat_start + i for i in range(rows)])
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # 创建图形
    plt.figure(figsize=(15, 6))
    
    # 绘制放大系数等值线图
    plt.subplot(1, 2, 1)
    contour = plt.contourf(lon_grid, lat_grid, m_values)
    plt.colorbar(contour, label='放大系数 (m)')
    plt.title('网格点的地图放大系数分布')
    plt.xlabel('经度 (°E)')
    plt.ylabel('纬度 (°N)')
    
    # 绘制科里奥利参数等值线图
    plt.subplot(1, 2, 2)
    contour = plt.contourf(lon_grid, lat_grid, f_values)
    plt.colorbar(contour, label='科里奥利参数 (f)')
    plt.title('网格点的科里奥利参数分布')
    plt.xlabel('经度 (°E)')
    plt.ylabel('纬度 (°N)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("计算极射投影（标准纬度70°N）下的网格点参数：")
    calculate_grid_parameters()