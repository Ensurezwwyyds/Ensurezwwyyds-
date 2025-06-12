import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 初始化参数
d = 100
phy0 = 5
a = 6371
omega = 7.292e-5
l_ref = a * np.cos(np.deg2rad(22.5)) * np.tan(np.deg2rad(phy0))

# 定义起始经纬度
lon_start = 100  # 起始经度（°E）
lat_start = 22.5  # 起始纬度（°N）

# 初始化矩阵
m = np.zeros((6, 7))
f = np.zeros((6, 7))

# 计算映射系数m和科氏力参数f
for In in range(7):
    for Jn in range(6):
        # 计算当前点的纬度（考虑到矩阵翻转）
        current_lat = lat_start - (5-Jn) * (d / (a * np.pi / 180))  # 5-Jn是因为矩阵会被翻转
        # 地图放大系数只与纬度有关
        m[Jn, In] = 1 / np.cos(np.deg2rad(current_lat))
        # 科氏力参数只与纬度有关
        f[Jn, In] = 2 * omega * np.sin(np.deg2rad(current_lat))

# 翻转矩阵
m = np.flipud(m)
f = np.flipud(f)

# 打印结果
print("映射系数m:")
print(m)
print("\n科氏力参数f:")
print(f)

# 将结果保存到Excel文件
df_m = pd.DataFrame(m)
df_f = pd.DataFrame(f)

df_m.to_excel('map_scale.xlsx', index=False, header=False)
df_f.to_excel('coriolis_force.xlsx', index=False, header=False)

# 计算经纬度网格
lon_start = 100  # 起始经度（°E）
lat_start = 22.5  # 起始纬度（°N）

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
plt.figure(figsize=(15, 6))

# 绘制地图放大系数的热力图
plt.subplot(121)
im1 = plt.imshow(m, cmap='viridis')
plt.colorbar(im1, label='地图放大系数')
plt.title('地图放大系数分布')

# 设置经纬度刻度
xticks = np.arange(7)
yticks = np.arange(6)
plt.xticks(xticks, [f'{lon_grid[0,i]:.1f}°E' for i in range(7)])
plt.yticks(yticks, [f'{lat_grid[i,0]:.1f}°N' for i in range(6)])
plt.xlabel('经度')
plt.ylabel('纬度')

# 绘制科氏力参数的热力图
plt.subplot(122)
im2 = plt.imshow(f, cmap='viridis')
plt.colorbar(im2, label='科氏力参数')
plt.title('科氏力参数分布')

# 设置经纬度刻度
plt.xticks(xticks, [f'{lon_grid[0,i]:.1f}°E' for i in range(7)])
plt.yticks(yticks, [f'{lat_grid[i,0]:.1f}°N' for i in range(6)])
plt.xlabel('经度')
plt.ylabel('纬度')

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()

# 打印经纬度网格
print("\n经度网格 (°E):")
print(lon_grid)
print("\n纬度网格 (°N):")
print(lat_grid)

print("\n结果已保存到Excel文件中")