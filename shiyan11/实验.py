# -*- coding: utf-8 -*-
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

# 读取数据
ds = xr.open_dataset("geo_197901.nc", decode_times=True)
z_geo = ds['z'].sel(pressure_level=500, valid_time="1979-01-10T00:00").values
ds.close()

z_a = z_geo / g
z_a = np.nan_to_num(z_a, nan=np.nanmean(z_a))

ds2 = xr.open_dataset("geo_197901.nc", decode_times=True)
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

dz_dy = np.zeros((M, N))
dz_dx = np.zeros((M, N))
for i in range(1, M - 1):
    for j in range(1, N - 1):
        dz_dy[i, j] = (z_a[i + 1, j] - z_a[i - 1, j]) / (2 * dy * dlat)
        dx_local = dy * np.cos(lat_rad[i]) * dlon
        dz_dx[i, j] = (z_a[i, j + 1] - z_a[i, j - 1]) / (2 * dx_local)

dz_dy[0, :] = dz_dy[1, :]
dz_dy[-1, :] = dz_dy[-2, :]
dz_dy[:, 0] = dz_dy[:, 1]
dz_dy[:, -1] = dz_dy[:, -2]
dz_dx[0, :] = dz_dx[1, :]
dz_dx[-1, :] = dz_dx[-2, :]
dz_dx[:, 0] = dz_dx[:, 1]
dz_dx[:, -1] = dz_dx[:, -2]

f_min = 1e-5
f = np.where(np.abs(f) < f_min, np.sign(f) * f_min, f)

# 在计算地转风时添加安全检查
# 改进地转风计算，避免极地附近的问题
U_b = np.zeros((M, N))
V_b = np.zeros((M, N))

# 只在中纬度地区计算地转风
for i in range(M):
    for j in range(N):
        # 限制在30°-70°纬度范围内计算地转风
        if 30 <= abs(lat_1d[i]) <= 70 and abs(f[i,j]) > 1e-10:
            U_b[i,j] = - (g / f[i,j]) * dz_dy[i,j]
            V_b[i,j] =   (g / f[i,j]) * dz_dx[i,j]
        else:
            U_b[i,j] = 0.0
            V_b[i,j] = 0.0

# 对地转风进行平滑处理
from scipy import ndimage
U_b = ndimage.gaussian_filter(U_b, sigma=1.0)
V_b = ndimage.gaussian_filter(V_b, sigma=1.0)

rm = np.ones((M, N))

def ti_total_energy_conservation(Ua, Va, z_a, U_b, V_b, z_b, rm, f, d, dt):
    M, N = z_a.shape
    U_c = np.zeros_like(U_b)
    V_c = np.zeros_like(V_b)
    z_c = np.zeros_like(z_b)

    ub = U_b * (rm / z_b)
    vb = V_b * (rm / z_b)
    c = 0.25 / d
    m1 = M - 1
    n1 = N - 1

    for i in range(1, m1):
        for j in range(1, n1):
            t1 = (U_b[i+1,j] + U_b[i,j]) * (ub[i+1,j] - ub[i,j]) \
               - (ub[i,j] + (U_b[i,j] + U_b[i-1,j])) * (ub[i,j] - ub[i-1,j])
            t2 = (V_b[i,j-1] + V_b[i,j]) * (ub[i,j] - ub[i,j-1]) \
               + (V_b[i,j] + V_b[i,j+1]) * (ub[i,j+1] - ub[i,j])
            t3 = 19.6 * z_b[i,j] * (z_b[i+1,j] - z_b[i-1,j])
            t4 = 2.0 * ub[i,j] * (U_b[i+1,j] - U_b[i-1,j]) \
               + 2.0 * ub[i,j] * (V_b[i,j+1] - V_b[i,j-1])
            e = - c * (rm[i,j]**2) * (t1 + t2 + t3 + t4) \
              + f[i,j] * z_b[i,j] * vb[i,j]
            U_c[i,j] = Ua[i,j] + e * dt

            t1g = (U_b[i+1,j] + U_b[i,j]) * (vb[i+1,j] - vb[i,j]) \
                - (U_b[i,j] + U_b[i-1,j]) * (vb[i,j] - vb[i-1,j])
            t2g = (V_b[i,j-1] + V_b[i,j]) * (vb[i,j] - vb[i,j-1]) \
                + (V_b[i,j] + V_b[i,j+1]) * (vb[i,j+1] - vb[i,j])
            t3g = 19.6 * z_b[i,j] * (z_b[i,j+1] - z_b[i,j-1])
            t4g = 2.0 * vb[i,j] * (U_b[i+1,j] - U_b[i-1,j]) \
                + 2.0 * vb[i,j] * (V_b[i,j+1] - V_b[i,j-1])
            g_term = - c * (rm[i,j]**2) * (t1g + t2g + t3g + t4g) \
                   - f[i,j] * z_b[i,j] * ub[i,j]
            V_c[i,j] = Va[i,j] + g_term * dt

    # 在计算过程中添加数值检查
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            # 检查输入值的有效性
            if not (np.isfinite(ub[i,j]) and np.isfinite(vb[i,j]) and 
                   np.isfinite(z_b[i,j]) and abs(f[i,j]) > 1e-10):
                continue
                
            h = -2.0 * c * (rm[i,j]**2) * (
                (U_b[i+1,j] - U_b[i-1,j]) - (V_b[i,j+1] - V_b[i,j-1])
            )
            # 检查计算结果的有效性
            if np.isfinite(e) and np.isfinite(g_term) and np.isfinite(h):
                U_c[i,j] = ub[i,j] + e * dt
                V_c[i,j] = vb[i,j] + g_term * dt
                z_c[i,j] = z_a[i,j] + h * dt
            else:
                # 如果计算结果无效，保持原值
                U_c[i,j] = ub[i,j]
                V_c[i,j] = vb[i,j]
                z_c[i,j] = z_a[i,j]
    
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

d_grid = dy * dlat
dt = 3600.0  # 保持1小时时间步长
# 修改积分时间为24小时
integration_hours = 24
num_steps = int(integration_hours * 3600 / dt)  # 积分24步

print(f"积分设置: {integration_hours}小时, 时间步长: {dt/3600:.1f}小时, 总步数: {num_steps}")

Ua = U_b.copy()
Va = V_b.copy()
z_prev = z_a.copy()

U_fore = Ua.copy()
V_fore = Va.copy()
z_fore = z_prev.copy()

# 添加诊断信息
print(f"\n初始场统计:")
print(f"z_a 范围: {np.nanmin(z_a):.2f} - {np.nanmax(z_a):.2f} m")
print(f"U_b 范围: {np.nanmin(U_b):.2f} - {np.nanmax(U_b):.2f} m/s")
print(f"V_b 范围: {np.nanmin(V_b):.2f} - {np.nanmax(V_b):.2f} m/s")

for step in tqdm(range(num_steps), desc=f"{integration_hours}h 积分", ncols=80):
    U_cn, V_cn, z_cn = ti_total_energy_conservation(
        Ua=U_fore, Va=V_fore,
        z_a=z_fore,
        U_b=U_b, V_b=V_b, z_b=z_a,
        rm=rm, f=f, d=d_grid, dt=dt
    )
    
    # 创建纬度掩码：赤道±20°内保持分析场
    lat_mask = np.abs(lat_1d) <= 20.0  # True表示赤道±20°内的区域
    
    # 在赤道±20°内保持分析场，其他区域使用积分结果
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
    if z_range > 10000:  # 如果高度场变化超过10km，可能有问题
        print(f"警告：第{step+1}步高度场变化过大: {z_range:.2f}m")
        break
    
    U_fore = U_cn.copy()
    V_fore = V_cn.copy()
    z_fore = z_cn.copy()
    
    print(f"第{step+1}步完成，z_c范围: {np.nanmin(z_cn):.2f} - {np.nanmax(z_cn):.2f} m")

z_c = z_fore.copy()

# 积分结束后的诊断
print(f"\n预报场统计:")
print(f"z_c 范围: {np.nanmin(z_c):.2f} - {np.nanmax(z_c):.2f} m")
print(f"有效值数量: {np.isfinite(z_c).sum()} / {z_c.size}")
print(f"NaN数量: {np.isnan(z_c).sum()}")
print(f"Inf数量: {np.isinf(z_c).sum()}")

# 计算RMSE
valid_mask = np.isfinite(z_c) & np.isfinite(z_a)
rmse = np.sqrt(np.nansum((z_c[valid_mask] - z_a[valid_mask])**2) / np.count_nonzero(valid_mask))
print(f"全场 RMSE = {rmse:.4f} m")

# 设置绘图的经纬度范围
latlim = [-20, 80]  # 纬度范围
lonlim = [60, 180]  # 经度范围

# 添加循环点用于绘图
z_a_cyclic, lon_cyclic = add_cyclic_point(z_a, coord=lon_1d)
z_c_cyclic, _ = add_cyclic_point(z_c, coord=lon_1d)

# 创建投影 - 调整中心经纬度以适应新的显示范围
proj = ccrs.LambertConformal(central_longitude=120.0, central_latitude=30.0, standard_parallels=(20.0, 60.0))

# 创建图形和子图
fig = plt.figure(figsize=(14, 6))  # 稍微增大图形尺寸

# 第一个子图：分析场
ax1 = fig.add_subplot(1, 2, 1, projection=proj)
ax1.set_title("分析场 (500 hPa 位势高度)")
ax1.set_extent([lonlim[0], lonlim[1], latlim[0], latlim[1]], crs=ccrs.PlateCarree())  # 设置显示范围
ax1.coastlines(resolution='50m')
ax1.gridlines(draw_labels=True, dms=True, linewidth=0.3)
lon2d, lat2d = np.meshgrid(lon_cyclic, lat_1d)
min_a, max_a = np.nanmin(z_a), np.nanmax(z_a)
levels_a = np.arange((min_a//150)*150, ((max_a//150)+1)*150, 150)
cf1 = ax1.contour(lon2d, lat2d, z_a_cyclic, levels=levels_a, colors='k', linewidths=1.0, transform=ccrs.PlateCarree())
ax1.clabel(cf1, inline=True, fmt="%d")

# 第二个子图：预报场
ax2 = fig.add_subplot(1, 2, 2, projection=proj)
ax2.set_title(f"{integration_hours}h 预报场 (500 hPa 位势高度)")  # 动态标题
ax2.set_extent([lonlim[0], lonlim[1], latlim[0], latlim[1]], crs=ccrs.PlateCarree())  # 设置显示范围
ax2.coastlines(resolution='50m')
ax2.gridlines(draw_labels=True, dms=True, linewidth=0.3)

# 更robust的等值线计算
valid_z_c = z_c[np.isfinite(z_c)]
if len(valid_z_c) > 0:
    min_c, max_c = np.nanmin(valid_z_c), np.nanmax(valid_z_c)
    print(f"预报场有效值范围: {min_c:.2f} - {max_c:.2f} m")
    
    # 使用与分析场相似的等值线间隔
    levels_c = levels_a.copy()
else:
    print("错误：预报场没有有效值")
    levels_c = np.arange(5000, 6000, 150)

# 绘制等值线时添加异常处理
try:
    cf2 = ax2.contour(lon2d, lat2d, z_c_cyclic, levels=levels_c, colors='k', linewidths=1.0, transform=ccrs.PlateCarree())
    ax2.clabel(cf2, inline=True, fmt="%d")
except Exception as e:
    print(f"绘制等值线时出错：{e}")
    ax2.text(0.5, 0.5, '数据异常\n无法绘制等值线', 
             transform=ax2.transAxes, ha='center', va='center', 
             fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat'))

plt.tight_layout()
plt.show()
