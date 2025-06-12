import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 初始化参数 - 中南半岛地区
d = 100  # 网格距离，单位km
phy0 = 15  # 标准纬度15°N（适合中南半岛地区）
a = 6371  # 地球半径，单位km
omega = 7.292e-5  # 地球角速度

# 计算参考长度，使用原点位置(100°E，20°N) - 中南半岛区域
lon_start = 100  # 起始经度（°E）
lat_start = 20   # 起始纬度（°N）
l_ref = a * np.cos(np.deg2rad(lat_start)) * np.tan(np.deg2rad(phy0))

# 初始化矩阵
m = np.zeros((6, 7))  # 地图放大系数矩阵
f = np.zeros((6, 7))  # 科氏力参数矩阵

# 计算映射系数m和科氏力参数f
for In in range(7):
    for Jn in range(6):
        # 计算当前点的纬度（考虑到矩阵翻转）
        current_lat = lat_start - (5-Jn) * (d / (a * np.pi / 180))  # 5-Jn是因为矩阵会被翻转
        # 地图放大系数只与纬度有关
        m[Jn, In] = 1 / np.cos(np.deg2rad(current_lat))
        # 科氏力参数只与纬度有关
        f[Jn, In] = 2 * omega * np.sin(np.deg2rad(current_lat))

# 翻转矩阵使得左下角为原点
m = np.flipud(m)
f = np.flipud(f)

# 打印结果
print("地图放大系数m (麦卡托投影，中南半岛区域):")
print(m)
print("\n科氏力参数f (麦卡托投影，中南半岛区域):")
print(f)

# 将结果保存到Excel文件
df_m = pd.DataFrame(m)
df_f = pd.DataFrame(f)

df_m.to_excel('map_scale_exp2.xlsx', index=False, header=False)
df_f.to_excel('coriolis_force_exp2.xlsx', index=False, header=False)

# 计算每个网格点对应的经纬度
lon_grid = np.zeros((6, 7))
lat_grid = np.zeros((6, 7))

for i in range(7):
    for j in range(6):
        # 计算经度变化（根据距离和纬度计算）
        delta_lon = (i * d) / (a * np.cos(np.deg2rad(lat_start)) * np.pi / 180)
        # 计算纬度变化（根据距离计算）
        delta_lat = (j * d) / (a * np.pi / 180)
        
        lon_grid[j, i] = lon_start + delta_lon
        lat_grid[j, i] = lat_start - delta_lat  # 纬度向南递减

# 翻转经纬度网格以匹配m和f矩阵
lon_grid = np.flipud(lon_grid)
lat_grid = np.flipud(lat_grid)

# 创建图形和子图
plt.figure(figsize=(15, 12))

# 绘制地图放大系数的热力图
plt.subplot(221)
im1 = plt.imshow(m, cmap='viridis')
plt.colorbar(im1, label='地图放大系数')
plt.title('麦卡托投影地图放大系数分布 (中南半岛区域)')

# 设置经纬度刻度
xticks = np.arange(7)
yticks = np.arange(6)
plt.xticks(xticks, [f'{lon_grid[0,i]:.1f}°E' for i in range(7)])
plt.yticks(yticks, [f'{lat_grid[i,0]:.1f}°N' for i in range(6)])
plt.xlabel('经度')
plt.ylabel('纬度')

# 绘制科氏力参数的热力图
plt.subplot(222)
im2 = plt.imshow(f, cmap='viridis')
plt.colorbar(im2, label='科氏力参数')
plt.title('麦卡托投影科氏力参数分布 (中南半岛区域)')

# 设置经纬度刻度
plt.xticks(xticks, [f'{lon_grid[0,i]:.1f}°E' for i in range(7)])
plt.yticks(yticks, [f'{lat_grid[i,0]:.1f}°N' for i in range(6)])
plt.xlabel('经度')
plt.ylabel('纬度')

# 计算WPS设置下的地图放大系数和科氏力参数
# 使用c.m文件中的公式进行计算
m_wps = np.zeros((6, 7))
f_wps = np.zeros((6, 7))

for In in range(7):
    for Jn in range(6):
        # 根据c.m文件中的公式计算
        m_wps[Jn, In] = np.sqrt((a * np.cos(np.deg2rad(lat_start)))**2 + (Jn * d)**2) / a
        f_wps[Jn, In] = 2 * omega * np.sin(Jn * d / np.sqrt((a * np.cos(np.deg2rad(lat_start)))**2 + (Jn * d)**2))

# 翻转矩阵
m_wps = np.flipud(m_wps)
f_wps = np.flipud(f_wps)

# 绘制WPS设置下的地图放大系数热力图
plt.subplot(223)
im3 = plt.imshow(m_wps, cmap='viridis')
plt.colorbar(im3, label='地图放大系数 (WPS)')
plt.title('WPS设置下的地图放大系数分布')

# 设置经纬度刻度
plt.xticks(xticks, [f'{lon_grid[0,i]:.1f}°E' for i in range(7)])
plt.yticks(yticks, [f'{lat_grid[i,0]:.1f}°N' for i in range(6)])
plt.xlabel('经度')
plt.ylabel('纬度')

# 绘制WPS设置下的科氏力参数热力图
plt.subplot(224)
im4 = plt.imshow(f_wps, cmap='viridis')
plt.colorbar(im4, label='科氏力参数 (WPS)')
plt.title('WPS设置下的科氏力参数分布')

# 设置经纬度刻度
plt.xticks(xticks, [f'{lon_grid[0,i]:.1f}°E' for i in range(7)])
plt.yticks(yticks, [f'{lat_grid[i,0]:.1f}°N' for i in range(6)])
plt.xlabel('经度')
plt.ylabel('纬度')

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()

# 计算差异
m_diff = m - m_wps
f_diff = f - f_wps

# 打印差异结果
print("\n地图放大系数差异 (自定义 - WPS):")
print(m_diff)
print("\n科氏力参数差异 (自定义 - WPS):")
print(f_diff)

# 打印经纬度网格
print("\n经度网格 (°E):")
print(lon_grid)
print("\n纬度网格 (°N):")
print(lat_grid)

print("\n结果已保存到Excel文件中")