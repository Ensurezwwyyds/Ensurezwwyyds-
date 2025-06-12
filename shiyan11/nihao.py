#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单文件脚本：读取 geo_197901.nc，做 12 小时积分预报，
在每步后做空间平滑，并在最终结果上做三点时间平滑 + 空间平滑，
按参考样式绘制初始场与平滑后预报场的 500 hPa 重力位势高度
"""

import os
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

# — 中文与负号配置 —
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def cmf(d, clat, clon, m, n):
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
    ua, va, za = ub.copy(), vb.copy(), zb.copy()
    ua[:,[0,n-1]] = ub[:,[0,n-1]]; va[:,[0,n-1]] = vb[:,[0,n-1]]; za[:,[0,n-1]] = zb[:,[0,n-1]]
    ua[[0,m-1],:] = ub[[0,m-1],:]; va[[0,m-1],:] = vb[[0,m-1],:]; za[[0,m-1],:] = zb[[0,m-1],:]
    return ua, va, za

def ti(ua, va, za, ub, vb, zb, rm, f, d, dt, zo, m, n):
    c = 0.25/d
    uc, vc, zc = np.zeros_like(ua), np.zeros_like(va), np.zeros_like(za)
    m1, n1 = m-1, n-1
    for i in range(1,m1):
        for j in range(1,n1):
            e = (-c*rm[i,j]*(
                  (ub[i+1,j]+ub[i,j])*(ub[i+1,j]-ub[i,j]) +
                  (ub[i,j]+ub[i-1,j])*(ub[i,j]-ub[i-1,j]) +
                  (vb[i,j-1]+vb[i,j])*(ub[i,j]-ub[i,j-1]) +
                  (vb[i,j]+vb[i,j+1])*(ub[i,j+1]-ub[i,j]) +
                  19.6*(zb[i+1,j]-zb[i-1,j])
                ) + f[i,j]*vb[i,j])
            uc[i,j] = ua[i,j] + e*dt
            g = (-c*rm[i,j]*(
                  (ub[i+1,j]+ub[i,j])*(vb[i+1,j]-vb[i,j]) +
                  (ub[i,j]+ub[i-1,j])*(vb[i,j]-vb[i-1,j]) +
                  (vb[i,j-1]+vb[i,j])*(vb[i,j]-vb[i,j-1]) +
                  (vb[i,j]+vb[i,j+1])*(vb[i,j+1]-vb[i,j]) +
                  19.6*(zb[i,j+1]-zb[i,j-1])
                ) - f[i,j]*ub[i,j])
            vc[i,j] = va[i,j] + g*dt
    for i in range(1,m1):
        for j in range(1,n1):
            h = (-c*rm[i,j]**2*(
                  (ub[i+1,j]+ub[i,j])*(zb[i+1,j]/rm[i+1,j]-zb[i,j]/rm[i,j]) +
                  (ub[i,j]+ub[i-1,j])*(zb[i,j]/rm[i,j]-zb[i-1,j]/rm[i-1,j]) +
                  (vb[i,j-1]+vb[i,j])*(zb[i,j]/rm[i,j]-zb[i,j-1]/rm[i,j-1]) +
                  (vb[i,j]+vb[i,j+1])*(zb[i,j+1]/rm[i,j+1]-zb[i,j]/rm[i,j]) +
                  2*(zb[i,j]-zo)/rm[i,j]*
                  (ub[i+1,j]-ub[i-1,j]+vb[i,j+1]-vb[i,j-1])
                ))
            zc[i,j] = za[i,j] + h*dt
    return uc, vc, zc

def ssbp(a, s, m, n):
    w = a.copy()
    m1, n1 = m-1, n-1
    for i in range(1,m1):
        for j in (1,n1-1):
            w[i,j] = (a[i,j]
                      +0.5*s*(1-s)*(a[i-1,j]+a[i+1,j]+a[i,j-1]+a[i,j+1]-4*a[i,j])
                      +0.25*s*s*(a[i-1,j-1]+a[i-1,j+1]+a[i+1,j-1]+a[i+1,j+1]-4*a[i,j]))
    for i in (1,m1-1):
        for j in range(1,n1):
            w[i,j] = (a[i,j]
                      +0.5*s*(1-s)*(a[i-1,j]+a[i+1,j]+a[i,j-1]+a[i,j+1]-4*a[i,j])
                      +0.25*s*s*(a[i-1,j-1]+a[i-1,j+1]+a[i+1,j-1]+a[i+1,j+1]-4*a[i,j]))
    return w

def time_smooth(za, zb, zc, s=0.5):
    zb2 = zb.copy()
    zb2[1:-1,1:-1] = zb[1:-1,1:-1] + s*(za[1:-1,1:-1] + zc[1:-1,1:-1] - 2*zb[1:-1,1:-1])/2
    return zb2

def plot_field(ax, data, lmda, phai, label):
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

    # 参数
    m,n = 41,17
    d, clat, clon = 300000.0, 45.0, 120.0
    dt, zo, s = 150.0, 0.0, 0.5

    # 投影、静力初始化
    rm,f,lmda,phai = cmf(d,clat,clon,m,n)
    ua,va,za = interp_proj_grid(u0,v0,z0,lmda,phai,m,n,lon,lat)
    ua,va    = cgw(za,rm,f,d,m,n)
    za0 = ssbp(za.copy(),s,m,n)

    # 边界准备
    ub,vb,zb = tbv(ua,va,za,m,n)
    uc,vc,zc = tbv(ua,va,za,m,n)

    # 积分（6 步 =12h），每步空间平滑，记录中点
    zb_mid = None
    for k in tqdm(range(6), desc="积分进度"):
        # 前半步
        tu,tv,tz = ti(ua,va,za, ua,va,za, rm,f,d,dt,zo,m,n)
        ub[1:-1,1:-1]=tu[1:-1,1:-1]; vb[1:-1,1:-1]=tv[1:-1,1:-1]; zb[1:-1,1:-1]=tz[1:-1,1:-1]
        ub,vb,zb = tbv(ub,vb,zb,m,n)
        ub = ssbp(ub,s,m,n); vb = ssbp(vb,s,m,n); zb = ssbp(zb,s,m,n)
        # 后半步
        tu,tv,tz = ti(ua,va,za, ub,vb,zb, rm,f,d,dt,zo,m,n)
        ua[1:-1,1:-1]=tu[1:-1,1:-1]; va[1:-1,1:-1]=tv[1:-1,1:-1]; za[1:-1,1:-1]=tz[1:-1,1:-1]
        ua,va,za = tbv(ua,va,za,m,n)
        ua = ssbp(ua,s,m,n); va = ssbp(va,s,m,n); za = ssbp(za,s,m,n)

        if k==2:
            zb_mid = za.copy()

    za12 = za

    # 时间平滑 + 再空间平滑
    zb_ts = time_smooth(za0, zb_mid, za12, s)
    zb_ts = ssbp(zb_ts, s, m, n)

    # 绘图
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,6),
        subplot_kw={'projection': ccrs.LambertConformal(
            central_longitude=clon, central_latitude=clat,
            standard_parallels=(30,60))})
    plot_field(ax1, za0,    lmda, phai, '(a) 原始场')
    plot_field(ax2, zb_ts,  lmda, phai, '(b) 时间+空间平滑后 12h 场')
    plt.suptitle('1979年1月10日 500hPa 重力位势高度场', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    main()
