import numpy as np
import matplotlib.pyplot as plt
from 实习内容 import magnification_factor_and_coriolis_parameter

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def calculate_parameters_for_latitudes():
    # 设置纬度值
    latitudes = [50, 60, 70, 80]
    
    # 设置网格参数
    d = 100  # 网格间距
    proj = 'stereographic'
    
    # 存储计算结果
    m_values = []
    f_values = []
    
    # 为每个纬度计算参数
    for lat in latitudes:
        In = lat
        Jn = lat
        
        # 计算放大系数和科里奥利参数
        m, f = magnification_factor_and_coriolis_parameter(proj, In, Jn, d)
        m_values.append(m)
        f_values.append(f)
        
        print(f"\n纬度 {lat}°N 的计算结果：")
        print(f"放大系数 (m): {m:.6f}")
        print(f"科里奥利参数 (f): {f:.6e}")
    
    # 创建图形
    plt.figure(figsize=(12, 5))
    
    # 绘制放大系数
    plt.subplot(1, 2, 1)
    plt.plot(latitudes, m_values, 'bo-', linewidth=2)
    plt.title('不同纬度的地图放大系数')
    plt.xlabel('纬度 (°N)')
    plt.ylabel('放大系数 (m)')
    plt.grid(True)
    
    # 绘制科里奥利参数
    plt.subplot(1, 2, 2)
    plt.plot(latitudes, f_values, 'ro-', linewidth=2)
    plt.title('不同纬度的科里奥利参数')
    plt.xlabel('纬度 (°N)')
    plt.ylabel('科里奥利参数 (f)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("计算极射赤面投影下不同纬度的地图放大系数和科里奥利参数：")
    calculate_parameters_for_latitudes()