# -*- coding: utf-8 -*-
"""
Runge-Kutta 4阶积分方案实现正压原始方程模式
基于MATLAB代码solve.f90和integration.f90的Python实现
"""

import os
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

# 中文与负号配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def cmf(d, clat, clon, m, n):
    """兰伯特投影坐标变换"""
    R = 6371000.0
    ω = 2*np.pi/(23*3600+56*60+4)
    θ = 30.0
    kk = ((np.log(np.sin(np.deg2rad(30))) - np.log(np.sin(np.deg2rad(60)))) /
          (np.log(np.tan(np.deg2rad(30/2))) - np.log(np.tan(np.deg2rad(60/2)))))
    le = R*np.sin(np.deg2rad(θ))/kk*(1/np.tan(np.deg2rad(θ/2)))**kk
    l1 = le*(np.tan(np.deg2rad(clat/2)))**kk

    rm   = np.zeros((m,n))
    f    = np.zeros((m,n))
    lmda = np.zeros((m,n))
    phai = np.zeros((m,n))

    for i in range(m):
        for j in range(n):
            II = i-(m-1)/2
            JJ = l1/d + (n-1)/2 - j
            L  = np.hypot(II,JJ)*d
            lt = (le**(2/kk) - L**(2/kk))/(le**(2/kk) + L**(2/kk))
            φ = np.rad2deg(np.arcsin(lt))
            λ = np.rad2deg(np.arctan2(II,JJ))/kk + clon
            rm[i,j] = (np.sin(np.deg2rad(θ))
                       /np.sin(np.deg2rad(90-φ))
                       *(np.tan(np.deg2rad((90-φ)/2))
                         /np.tan(np.deg2rad(θ/2)))**kk)
            f[i,j]  = 2*ω*np.sin(np.deg2rad(φ))
            lmda[i,j], phai[i,j] = λ, φ

    return rm, f, lmda, phai

def cgw(za, rm, f, d, m, n):
    """地转风计算"""
    c = 9.8/d
    ua = np.zeros((m,n)); va = np.zeros((m,n))
    for i in range(m):
        ua[i,0]   = -c*rm[i,0]*(za[i,1]-za[i,0])/f[i,0]
        ua[i,n-1] = -c*rm[i,n-1]*(za[i,n-1]-za[i,n-2])/f[i,n-1]
        for j in range(1,n-1):
            ua[i,j] = -c*rm[i,j]*(za[i,j+1]-za[i,j])/f[i,j]
    for j in range(n):
        va[0,j]   =  c*rm[0,j]*(za[1,j]-za[0,j])/f[0,j]
        va[m-1,j] =  c*rm[m-1,j]*(za[m-1,j]-za[m-2,j])/f[m-1,j]
        for i in range(1,m-1):
            va[i,j] = c*rm[i,j]*(za[i+1,j]-za[i,j])/f[i,j]
    return ua, va

def interp_proj_grid(u, v, z, lmda, phai, m, n, lon, lat):
    """插值到投影网格"""
    Lon, Lat = np.meshgrid(lon, lat)
    pts_src = np.column_stack((Lon.ravel(), Lat.ravel()))
    pts_tgt = np.column_stack((lmda.ravel(), phai.ravel()))

    ui_lin = griddata(pts_src, u.ravel(), pts_tgt, method='linear')
    ui_nn  = griddata(pts_src, u.ravel(), pts_tgt, method='nearest')
    vi_lin = griddata(pts_src, v.ravel(), pts_tgt, method='linear')
    vi_nn  = griddata(pts_src, v.ravel(), pts_tgt, method='nearest')
    zi_lin = griddata(pts_src, z.ravel(), pts_tgt, method='linear')
    zi_nn  = griddata(pts_src, z.ravel(), pts_tgt, method='nearest')

    ui = ui_lin.copy(); vi = vi_lin.copy(); zi = zi_lin.copy()
    mask = np.isnan(ui); ui[mask] = ui_nn[mask]
    mask = np.isnan(vi); vi[mask] = vi_nn[mask]
    mask = np.isnan(zi); zi[mask] = zi_nn[mask]

    return ui.reshape(m,n), vi.reshape(m,n), zi.reshape(m,n)

def tbv(ub, vb, zb, m, n):
    """边界条件处理"""
    ua, va, za = ub.copy(), vb.copy(), zb.copy()
    ua[:,[0,n-1]] = ub[:,[0,n-1]]; va[:,[0,n-1]] = vb[:,[0,n-1]]; za[:,[0,n-1]] = zb[:,[0,n-1]]
    ua[[0,m-1],:] = ub[[0,m-1],:]; va[[0,m-1],:] = vb[[0,m-1],:]; za[[0,m-1],:] = zb[[0,m-1],:]
    return ua, va, za

def compute_tendency(u, v, z, rm, f, d, zo, m, n):
    """计算时间倾向项 (基于正压原始方程)"""
    c = 0.25/d
    u_t = np.zeros_like(u)
    v_t = np.zeros_like(v)
    z_t = np.zeros_like(z)
    
    m1, n1 = m-1, n-1
    
    # u方程时间倾向
    for i in range(1, m1):
        for j in range(1, n1):
            # 平流项
            advection_u = (-c * rm[i,j] * (
                (u[i+1,j] + u[i,j]) * (u[i+1,j] - u[i,j]) +
                (u[i,j] + u[i-1,j]) * (u[i,j] - u[i-1,j]) +
                (v[i,j-1] + v[i,j]) * (u[i,j] - u[i,j-1]) +
                (v[i,j] + v[i,j+1]) * (u[i,j+1] - u[i,j]) +
                19.6 * (z[i+1,j] - z[i-1,j])
            ))
            # 科里奥利力项
            coriolis_u = f[i,j] * v[i,j]
            u_t[i,j] = advection_u + coriolis_u
    
    # v方程时间倾向
    for i in range(1, m1):
        for j in range(1, n1):
            # 平流项
            advection_v = (-c * rm[i,j] * (
                (u[i+1,j] + u[i,j]) * (v[i+1,j] - v[i,j]) +
                (u[i,j] + u[i-1,j]) * (v[i,j] - v[i-1,j]) +
                (v[i,j-1] + v[i,j]) * (v[i,j] - v[i,j-1]) +
                (v[i,j] + v[i,j+1]) * (v[i,j+1] - v[i,j]) +
                19.6 * (z[i,j+1] - z[i,j-1])
            ))
            # 科里奥利力项
            coriolis_v = -f[i,j] * u[i,j]
            v_t[i,j] = advection_v + coriolis_v
    
    # z方程时间倾向 (连续方程)
    for i in range(1, m1):
        for j in range(1, n1):
            # 散度项
            divergence = (-c * rm[i,j]**2 * (
                (u[i+1,j] + u[i,j]) * (z[i+1,j]/rm[i+1,j] - z[i,j]/rm[i,j]) +
                (u[i,j] + u[i-1,j]) * (z[i,j]/rm[i,j] - z[i-1,j]/rm[i-1,j]) +
                (v[i,j-1] + v[i,j]) * (z[i,j]/rm[i,j] - z[i,j-1]/rm[i,j-1]) +
                (v[i,j] + v[i,j+1]) * (z[i,j+1]/rm[i,j+1] - z[i,j]/rm[i,j]) +
                2 * (z[i,j] - zo) / rm[i,j] *
                (u[i+1,j] - u[i-1,j] + v[i,j+1] - v[i,j-1])
            ))
            z_t[i,j] = divergence
    
    return u_t, v_t, z_t

def runge_kutta_4(u, v, z, rm, f, d, dt, zo, m, n):
    """Runge-Kutta 4阶积分方案
    
    根据公式:
    y_{n+1} = y_n + (1/6)(k1 + 2*k2 + 2*k3 + k4)
    
    其中:
    k1 = h*f(x_n, y_n)
    k2 = h*f(x_n + h/2, y_n + k1/2)
    k3 = h*f(x_n + h/2, y_n + k2/2)
    k4 = h*f(x_n + h, y_n + k3)
    """
    
    # 计算 k1
    k1_u, k1_v, k1_z = compute_tendency(u, v, z, rm, f, d, zo, m, n)
    k1_u *= dt
    k1_v *= dt
    k1_z *= dt
    
    # 计算 k2
    u_temp = u + 0.5 * k1_u
    v_temp = v + 0.5 * k1_v
    z_temp = z + 0.5 * k1_z
    u_temp, v_temp, z_temp = tbv(u_temp, v_temp, z_temp, m, n)
    
    k2_u, k2_v, k2_z = compute_tendency(u_temp, v_temp, z_temp, rm, f, d, zo, m, n)
    k2_u *= dt
    k2_v *= dt
    k2_z *= dt
    
    # 计算 k3
    u_temp = u + 0.5 * k2_u
    v_temp = v + 0.5 * k2_v
    z_temp = z + 0.5 * k2_z
    u_temp, v_temp, z_temp = tbv(u_temp, v_temp, z_temp, m, n)
    
    k3_u, k3_v, k3_z = compute_tendency(u_temp, v_temp, z_temp, rm, f, d, zo, m, n)
    k3_u *= dt
    k3_v *= dt
    k3_z *= dt
    
    # 计算 k4
    u_temp = u + k3_u
    v_temp = v + k3_v
    z_temp = z + k3_z
    u_temp, v_temp, z_temp = tbv(u_temp, v_temp, z_temp, m, n)
    
    k4_u, k4_v, k4_z = compute_tendency(u_temp, v_temp, z_temp, rm, f, d, zo, m, n)
    k4_u *= dt
    k4_v *= dt
    k4_z *= dt
    
    # RK4最终结果
    u_new = u + (k1_u + 2*k2_u + 2*k3_u + k4_u) / 6.0
    v_new = v + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6.0
    z_new = z + (k1_z + 2*k2_z + 2*k3_z + k4_z) / 6.0
    
    return u_new, v_new, z_new

def ssbp(a, s, m, n):
    """空间平滑"""
    w = a.copy()
    m1, n1 = m-1, n-1
    for i in range(1,m1):
        for j in range(1,n1):
            w[i,j] = (a[i,j]
                      +0.5*s*(1-s)*(a[i-1,j]+a[i+1,j]+a[i,j-1]+a[i,j+1]-4*a[i,j])
                      +0.25*s*s*(a[i-1,j-1]+a[i-1,j+1]+a[i+1,j-1]+a[i+1,j+1]-4*a[i,j]))
    return w

def time_smooth(za, zb, zc, s=0.5):
    """时间平滑"""
    zb2 = zb.copy()
    zb2[1:-1,1:-1] = zb[1:-1,1:-1] + s*(za[1:-1,1:-1] + zc[1:-1,1:-1] - 2*zb[1:-1,1:-1])/2
    return zb2

def plot_field(ax, data, lmda, phai, label):
    """绘制等高线图"""
    ax.set_extent([60,180,20,80], crs=ccrs.PlateCarree())
    ax.coastlines('50m', linewidth=0.8)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linestyle='--', linewidth=0.5, color='gray')
    gl.top_labels = False; gl.right_labels = False
    gl.xlocator = plt.FixedLocator(np.arange(60,181,30))
    gl.ylocator = plt.FixedLocator(np.arange(20,81,10))
    levels = np.arange(5000,5751,150)
    cs = ax.contour(lmda, phai, data, levels=levels,
                    colors='black', linewidths=1.2,
                    transform=ccrs.PlateCarree())
    ax.clabel(cs, fmt='%d', inline=True, fontsize=8)
    ax.set_title(label, fontsize=12, pad=10)

def main():
    """主函数 - 使用RK4积分方案的正压原始方程模式"""
    # 读取数据
    nc = 'geo_197901.nc'
    if not os.path.isfile(nc):
        raise FileNotFoundError(f"找不到数据文件: {nc}")

    ds = xr.open_dataset(nc, decode_times=False)
    if 'pressure_level' in ds.dims:
        ds = ds.sel(pressure_level=500)
    z500 = ds['z']/9.8
    u500 = ds['u']; v500 = ds['v']
    lon = ds.longitude.values; lat = ds.latitude.values

    # 初始场 t=0
    nt0 = 0
    u0 = u500.isel(valid_time=nt0).values
    v0 = v500.isel(valid_time=nt0).values
    z0 = z500.isel(valid_time=nt0).values

    # 模式参数
    m, n = 41, 17  # 网格点数
    d, clat, clon = 300000.0, 45.0, 120.0  # 网格距、中心纬度、中心经度
    dt, zo, s = 150.0, 0.0, 0.5  # 时间步长、参考高度、平滑系数
    
    print("正在初始化模式...")
    
    # 投影坐标变换和静力初始化
    rm, f, lmda, phai = cmf(d, clat, clon, m, n)
    ua, va, za = interp_proj_grid(u0, v0, z0, lmda, phai, m, n, lon, lat)
    ua, va = cgw(za, rm, f, d, m, n)  # 地转风初始化
    za0 = ssbp(za.copy(), s, m, n)  # 初始场空间平滑

    # 边界条件初始化
    ub, vb, zb = tbv(ua, va, za, m, n)
    
    print("开始RK4积分...")
    
    # 使用RK4进行时间积分 (6步 = 12小时)
    n_steps = 6
    zb_mid = None
    
    for k in tqdm(range(n_steps), desc="RK4积分进度"):
        # 使用RK4方案进行一个时间步的积分
        ub_new, vb_new, zb_new = runge_kutta_4(ub, vb, zb, rm, f, d, dt, zo, m, n)
        
        # 更新变量
        ub[1:-1, 1:-1] = ub_new[1:-1, 1:-1]
        vb[1:-1, 1:-1] = vb_new[1:-1, 1:-1]
        zb[1:-1, 1:-1] = zb_new[1:-1, 1:-1]
        
        # 边界条件处理
        ub, vb, zb = tbv(ub, vb, zb, m, n)
        
        # 空间平滑
        ub = ssbp(ub, s, m, n)
        vb = ssbp(vb, s, m, n)
        zb = ssbp(zb, s, m, n)
        
        # 记录中间时刻的场
        if k == 2:  # 第3步，即6小时
            zb_mid = zb.copy()
    
    za12 = zb  # 12小时预报场
    
    print("进行时间和空间平滑...")
    
    # 时间平滑 + 再次空间平滑
    zb_ts = time_smooth(za0, zb_mid, za12, s)
    zb_ts = ssbp(zb_ts, s, m, n)
    
    print("绘制结果...")
    
    # 绘制对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
        subplot_kw={'projection': ccrs.LambertConformal(
            central_longitude=clon, central_latitude=clat,
            standard_parallels=(30, 60))})
    
    plot_field(ax1, za0, lmda, phai, '(a) 初始场 (RK4)')
    plot_field(ax2, zb_ts, lmda, phai, '(b) RK4积分12h预报场')
    
    plt.suptitle('RK4积分方案 - 1979年1月10日 500hPa 重力位势高度场', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    
    print("RK4积分完成！")
    
    # 输出一些统计信息
    print(f"初始场最大值: {np.max(za0):.2f} m")
    print(f"初始场最小值: {np.min(za0):.2f} m")
    print(f"12h预报场最大值: {np.max(zb_ts):.2f} m")
    print(f"12h预报场最小值: {np.min(zb_ts):.2f} m")
    print(f"最大变化: {np.max(np.abs(zb_ts - za0)):.2f} m")

if __name__ == '__main__':
    main()