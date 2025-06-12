# 习题一：使用geo.19790602.nc进行24小时天气预报
# 分析预报初值对正压原始方程模式预报效果的影响

import numpy as np
import xarray as xr
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

g = 9.8
Omega = 7.2921159e-5

# 读取数据 - 修改为geo.19790602.nc，时间为1979年6月2日00时
ds = xr.open_dataset("geo.19790602.nc", decode_times=True)
z_geo = ds['z'].sel(pressure_level=500, valid_time="1979-06-02T00:00").values
ds.close()

z_a = z_geo / g
z_a = np.nan_to_num(z_a, nan=np.nanmean(z_a))

ds2 = xr.open_dataset("geo.19790602.nc", decode_times=True)
lat_1d = ds2['latitude'].values
lon_1d = ds2['longitude'].values
ds2.close()

M, N = z_a.shape
lat_rad = np.deg2rad(lat_1d)
f_1d = 2 * Omega * np.sin(lat_rad)
f = np.repeat(f_1d[:, np.newaxis], N, axis=1)

R = 6371e3
dy = (np.pi / 180.0) * R
dlat = abs(lat_1d[1] - lat_1d[0])
dlon = abs(lon_1d[1] - lon_1d[0])

# 计算地转风
dz_dy = np.zeros((M, N))
dz_dx = np.zeros((M, N))
for i in range(1, M - 1):
    for j in range(1, N - 1):
        dz_dy[i, j] = (z_a[i + 1, j] - z_a[i - 1, j]) / (2 * dy * dlat)
        dx_local = dy * np.cos(lat_rad[i]) * dlon
        dz_dx[i, j] = (z_a[i, j + 1] - z_a[i, j - 1]) / (2 * dx_local)

# 边界处理
dz_dy[0, :] = dz_dy[1, :]
dz_dy[-1, :] = dz_dy[-2, :]
dz_dy[:, 0] = dz_dy[:, 1]
dz_dy[:, -1] = dz_dy[:, -2]
dz_dx[0, :] = dz_dx[1, :]
dz_dx[-1, :] = dz_dx[-2, :]
dz_dx[:, 0] = dz_dx[:, 1]
dz_dx[:, -1] = dz_dx[:, -2]

# 地转风计算（添加安全检查）
U_b = np.zeros((M, N))
V_b = np.zeros((M, N))

for i in range(M):
    for j in range(N):
        if 30 <= abs(lat_1d[i]) <= 70 and abs(f[i,j]) > 1e-10:
            U_b[i, j] = -g * dz_dy[i, j] / f[i, j]
            V_b[i, j] = g * dz_dx[i, j] / f[i, j]
        else:
            # 在低纬度或高纬度地区使用梯度风近似
            U_b[i, j] = -g * dz_dy[i, j] / (1e-4 if abs(f[i,j]) < 1e-10 else f[i, j])
            V_b[i, j] = g * dz_dx[i, j] / (1e-4 if abs(f[i,j]) < 1e-10 else f[i, j])

# 限制风速范围
U_b = np.clip(U_b, -100, 100)
V_b = np.clip(V_b, -100, 100)

# 计算地图因子
rm = np.zeros((M, N))
for i in range(M):
    rm[i, :] = R * np.cos(lat_rad[i])

# 总能量守恒的时间积分函数
def ti_total_energy_conservation(Ua, Va, z_a, U_b, V_b, z_b, rm, f, d, dt):
    M, N = Ua.shape
    U_c = np.zeros((M, N))
    V_c = np.zeros((M, N))
    z_c = np.zeros((M, N))
    
    # 计算散度
    div_UV = np.zeros((M, N))
    for i in range(1, M-1):
        for j in range(1, N-1):
            du_dx = (Ua[i, j+1] - Ua[i, j-1]) / (2 * d * np.cos(np.deg2rad(lat_1d[i])))
            dv_dy = (Va[i+1, j] - Va[i-1, j]) / (2 * d)
            div_UV[i, j] = du_dx + dv_dy
    
    # 边界处理
    div_UV[0, :] = div_UV[1, :]
    div_UV[-1, :] = div_UV[-2, :]
    div_UV[:, 0] = div_UV[:, 1]
    div_UV[:, -1] = div_UV[:, -2]
    
    # 时间积分
    for i in range(M):
        for j in range(N):
            # 添加数值稳定性检查
            if abs(f[i, j]) > 1e-10:
                U_c[i, j] = Ua[i, j] + dt * (f[i, j] * Va[i, j] - g * (z_a[i+1, j] - z_a[i-1, j]) / (2 * d) if i > 0 and i < M-1 else 0)
                V_c[i, j] = Va[i, j] + dt * (-f[i, j] * Ua[i, j] - g * (z_a[i, j+1] - z_a[i, j-1]) / (2 * d * np.cos(np.deg2rad(lat_1d[i]))) if j > 0 and j < N-1 else 0)
            else:
                U_c[i, j] = Ua[i, j]
                V_c[i, j] = Va[i, j]
            
            z_c[i, j] = z_a[i, j] - dt * div_UV[i, j] # 使用 z_a (上一时刻的预报场) 作为基础
    
    # 边界条件处理
    U_c[0, :]  = U_c[1, :]
    U_c[-1, :] = U_c[-2, :]
    U_c[:, 0]  = U_c[:, 1]
    U_c[:, -1] = U_c[:, -2]
    V_c[0, :]  = V_c[1, :]
    V_c[-1, :] = V_c[-2, :]
    V_c[:, 0]  = V_c[:, 1]
    V_c[:, -1] = V_c[:, -2]
    z_c[0, :]  = z_c[1, :]
    z_c[-1, :] = z_c[-2, :]
    z_c[:, 0]  = z_c[:, 1]
    z_c[:, -1] = z_c[:, -2]

    return U_c, V_c, z_c

# 积分设置
d_grid = dy * dlat
dt = 3600.0  # 1小时时间步长
integration_hours = 24  # 24小时积分
num_steps = int(integration_hours * 3600 / dt)

print(f"习题一：1979年6月2日00时初始场24小时预报")
print(f"积分设置: {integration_hours}小时, 时间步长: {dt/3600:.1f}小时, 总步数: {num_steps}")

# 初始化预报场
U_fore = U_b.copy()
V_fore = V_b.copy()
z_fore = z_a.copy()

print(f"\n初始场统计:")
print(f"z_a 范围: {np.nanmin(z_a):.2f} - {np.nanmax(z_a):.2f} m")
print(f"U_b 范围: {np.nanmin(U_b):.2f} - {np.nanmax(U_b):.2f} m/s")
print(f"V_b 范围: {np.nanmin(V_b):.2f} - {np.nanmax(V_b):.2f} m/s")

# 时间积分循环
# 时间积分循环
for step in tqdm(range(num_steps), desc=f"{integration_hours}h 积分", ncols=80):
    U_cn, V_cn, z_cn = ti_total_energy_conservation(
        Ua=U_fore, Va=V_fore,
        z_a=z_fore,
        U_b=U_b, V_b=V_b, z_b=z_a,  # 使用初始场作为背景场
        rm=rm, f=f, d=d_grid, dt=dt
    )
    
    # 在赤道±20°范围内保持分析场
    lat_mask = np.abs(lat_1d) <= 20.0
    U_cn[lat_mask, :] = U_b[lat_mask, :]
    V_cn[lat_mask, :] = V_b[lat_mask, :]
    z_cn[lat_mask, :] = z_a[lat_mask, :]
    
    # 检查数值是否合理
    if np.isnan(z_cn).any() or np.isinf(z_cn).any():
        print(f"警告：第{step+1}步出现数值问题")
        print(f"NaN数量: {np.isnan(z_cn).sum()}")
        print(f"Inf数量: {np.isinf(z_cn).sum()}")
        break
    
    # 检查数值范围是否合理
    z_range = np.nanmax(z_cn) - np.nanmin(z_cn)
    if z_range > 10000:
        print(f"警告：第{step+1}步高度场变化过大: {z_range:.2f}m")
        break
    
    U_fore = U_cn.copy()
    V_fore = V_cn.copy()
    z_fore = z_cn.copy()
    
    if (step + 1) % 6 == 0:  # 每6小时输出一次
        print(f"第{step+1}步完成，z_c范围: {np.nanmin(z_cn):.2f} - {np.nanmax(z_cn):.2f} m")

z_c = z_fore.copy()

# 积分结束后的诊断
print(f"\n预报场统计:")
print(f"z_c 范围: {np.nanmin(z_c):.2f} - {np.nanmax(z_c):.2f} m")
print(f"有效值数量: {np.isfinite(z_c).sum()} / {z_c.size}")
print(f"NaN数量: {np.isnan(z_c).sum()}")
print(f"Inf数量: {np.isinf(z_c).sum()}")

# 计算RMSE
valid_mask = np.isfinite(z_a) & np.isfinite(z_c)
if np.sum(valid_mask) > 0:
    rmse = np.sqrt(np.mean((z_c[valid_mask] - z_a[valid_mask])**2))
    print(f"\nRMSE: {rmse:.2f} m")
else:
    print("\n无法计算RMSE：没有有效的对比数据")

# 绘图部分
z_a_cyclic, lon_cyclic = add_cyclic_point(z_a, coord=lon_1d)
z_c_cyclic, _ = add_cyclic_point(z_c, coord=lon_1d)

# 设置绘图的经纬度范围
latlim = [-20, 80]
lonlim = [60, 180]

# 创建投影
proj = ccrs.LambertConformal(central_longitude=120.0, central_latitude=30.0, standard_parallels=(20.0, 60.0))

# 创建图形
fig = plt.figure(figsize=(16, 8))
fig.suptitle('习题一：1979年6月2日初始场24小时预报结果对比', fontsize=16, fontweight='bold')

# 第一个子图：分析场
ax1 = fig.add_subplot(1, 2, 1, projection=proj)
ax1.set_title("分析场 (500 hPa 位势高度)")
ax1.set_extent([lonlim[0], lonlim[1], latlim[0], latlim[1]], crs=ccrs.PlateCarree())
ax1.coastlines(resolution='50m')
ax1.gridlines(draw_labels=True, dms=True, linewidth=0.3)

# 第一个子图：分析场
lon2d, lat2d = np.meshgrid(lon_cyclic, lat_1d)
min_a, max_a = np.nanmin(z_a), np.nanmax(z_a)
levels_a = np.arange((min_a//150)*150, ((max_a//150)+1)*150, 150)  # 修改为150间隔
cf1 = ax1.contour(lon2d, lat2d, z_a_cyclic, levels=levels_a, colors='k', linewidths=1.0, transform=ccrs.PlateCarree())
ax1.clabel(cf1, inline=True, fmt="%d")

# 第二个子图：预报场
ax2 = fig.add_subplot(1, 2, 2, projection=proj)
ax2.set_title(f"{integration_hours}h 预报场 (500 hPa 位势高度)")
ax2.set_extent([lonlim[0], lonlim[1], latlim[0], latlim[1]], crs=ccrs.PlateCarree())
ax2.coastlines(resolution='50m')
ax2.gridlines(draw_labels=True, dms=True, linewidth=0.3)

# 更robust的等值线计算
valid_z_c = z_c[np.isfinite(z_c)]
if len(valid_z_c) > 0:
    min_c, max_c = np.nanmin(valid_z_c), np.nanmax(valid_z_c)
    print(f"预报场有效值范围: {min_c:.2f} - {max_c:.2f} m")
    levels_c = levels_a.copy()
else:
    print("错误：预报场没有有效值")
    levels_c = np.arange(5000, 6000, 150)  # 修改为150间隔

# 绘制等值线
try:
    cf2 = ax2.contour(lon2d, lat2d, z_c_cyclic, levels=levels_c, colors='k', linewidths=1.0, transform=ccrs.PlateCarree())
    ax2.clabel(cf2, inline=True, fmt="%d")
except Exception as e:
    print(f"绘制等值线时出错：{e}")

plt.tight_layout()
plt.show()

# 输出分析结论
print("\n=== 习题一分析结论 ===")
print("1. 预报初值时间：1979年6月2日00时")
print("2. 预报时长：24小时")
print(f"3. 预报效果评估：RMSE = {rmse:.2f} m" if 'rmse' in locals() else "3. 预报效果评估：无法计算RMSE")
print("4. 预报初值对模式效果的影响：")
print("   - 不同的初始时刻会影响大气环流的初始状态")
print("   - 6月2日的大气状态可能与1月10日存在显著差异")
print("   - 季节性差异会影响预报的准确性和稳定性")
print("   - 建议与原始实验（1月10日）进行对比分析")