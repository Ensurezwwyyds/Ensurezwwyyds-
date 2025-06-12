import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 基本参数
EARTH_RADIUS = 6371  # 地球半径
OMEGA = 7.292e-5  # 地球自转角速度

def set_chinese_font():
    """设置matplotlib中文字体"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def set_parameters():
    """设置投影参数"""
    # 网格参数
    M = 10  # 东西扩展（考虑到东亚地区的经度范围较大）
    N = 8   # 南北扩展（考虑到纬度范围）
    d = 500  # 格点距（km）
    
    # 投影参数
    phy0 = 40  # 中央纬度（东亚地区的中心纬度）
    seita1 = 30  # 第一标准纬度
    seita2 = 50  # 第二标准纬度（调整以优化东亚地区的投影）
    a = EARTH_RADIUS  # 地球半径
    
    return M, N, d, phy0, seita1, seita2, a

def calculate_projection_factors(M, N, d, phy0, seita1, seita2, a):
    """计算投影因子"""
    # 计算圆锥常数
    k = (np.log(np.sin(np.deg2rad(seita1))) - np.log(np.sin(np.deg2rad(seita2)))) / \
        (np.log(np.tan(np.deg2rad(seita1/2))) - np.log(np.tan(np.deg2rad(seita2/2))))
    
    # 计算映像平面上赤道到北极点的距离
    le = a * np.sin(np.deg2rad(seita1)) / k * (1/np.tan(np.deg2rad(seita1/2)))**k
    
    # 计算参考距离
    l_ref = le * (np.cos(np.deg2rad(phy0))/(1+np.sin(np.deg2rad(phy0))))**k
    
    # 初始化放大系数矩阵
    m = np.zeros((2 * N + 1, 2 * M + 1))
    
    # 计算每个网格点的放大系数
    for In in range(-M, M+1):
        for Jn in range(-N, N+1):
            l = np.sqrt((abs(In) * d)**2 + (l_ref-Jn*d)**2)
            m[Jn+N, In+M] = k*l/(a*np.sqrt(1 - \
                ((le**(2/k) - l**(2/k))/(le**(2/k) + l**(2/k)))**2))
    
    return np.flipud(m)

def plot_magnification_factor(m):
    """绘制放大系数等值线图"""
    # 设置经纬度范围（东亚地区：100°~150°E，20°~60°N）
    lon_range = np.linspace(100, 150, 2*10+1)  # 经度范围
    lat_range = np.linspace(20, 60, 2*8+1)    # 纬度范围
    
    # 生成网格点坐标
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
    
    plt.figure(figsize=(12, 10))
    
    # 绘制等值线
    levels = np.linspace(np.min(m), np.max(m), 15)  # 增加等值线级别
    contour = plt.contour(lon_grid, lat_grid, m, levels=levels, colors='k')
    plt.clabel(contour, inline=True, fontsize=8)  # 添加等值线标签
    
    # 添加填充等值线
    contourf = plt.contourf(lon_grid, lat_grid, m, levels=levels, cmap='RdYlBu_r', alpha=0.6)
    plt.colorbar(contourf, label='放大系数')
    
    # 设置网格和标签
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title('东亚地区兰伯特投影放大系数分布', pad=15, fontsize=14)
    plt.xlabel('经度 (°)', fontsize=12)
    plt.ylabel('纬度 (°)', fontsize=12)
    
    # 调整显示范围
    plt.xlim(100, 150)
    plt.ylim(20, 60)
    
    # 设置刻度格式，强调东西南北方向
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'{x}°E'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'{x}°N'))
    
    plt.tight_layout()
    plt.show()

def save_to_excel(lon_range, lat_range, m, f):
    """保存结果到Excel文件"""
    # 创建结果数据
    data = []
    for i in range(len(lon_range)):
        for j in range(len(lat_range)):
            data.append({
                '经度': f'{lon_range[i]:.2f}°E',
                '纬度': f'{lat_range[j]:.2f}°N',
                '地图系数': f'{m[j,i]:.4f}',
                '科里奥利参数': f'{f[j,i]:.4e}'
            })
    
    # 创建DataFrame并保存
    df = pd.DataFrame(data)
    df.to_excel('习题二_地图系数.xlsx', index=False)
    df.to_excel('习题二_科里奥利参数.xlsx', index=False)

def main():
    # 设置中文字体
    set_chinese_font()
    
    # 设置参数
    M, N, d, phy0, seita1, seita2, a = set_parameters()
    
    # 计算放大系数
    m = calculate_projection_factors(M, N, d, phy0, seita1, seita2, a)
    
    # 计算经纬度范围
    lon_range = np.linspace(100, 150, 2*M+1)
    lat_range = np.linspace(20, 60, 2*N+1)
    
    # 计算科里奥利参数
    f = 2 * OMEGA * np.sin(np.deg2rad(lat_range))[:, np.newaxis] * np.ones_like(m)
    
    # 保存结果到Excel
    save_to_excel(lon_range, lat_range, m, f)
    
    # 绘制等值线图
    plot_magnification_factor(m)

if __name__ == '__main__':
    main()