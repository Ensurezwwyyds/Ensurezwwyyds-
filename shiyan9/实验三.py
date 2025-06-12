import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

def load_500hpa_height(nc_file):
    """
    读取 NetCDF，只加载 500 hPa 第一个时次的位势（z），
    并转换为高度场（m），返回 height(lat, lon)、lon、lat。
    """
    with xr.open_dataset(nc_file, engine='netcdf4', decode_times=False) as ds:
        # 先选层，再按位置选第一个时次，避免 KeyError
        z0 = ds['z'].sel(pressure_level=500).isel(valid_time=0)
        height = (z0 / 9.80665).values  # 转换为 m

        # 经度转换到 [-180,180) 并排序
        lon = ((ds.longitude.values + 180) % 360) - 180
        idx = np.argsort(lon)
        lon = lon[idx]
        height = height[:, idx]

        lat = ds.latitude.values

    return height, lon, lat

def smooth_9point(data):
    """
    高效九点平滑（分离卷积）：
    1) 纬度方向边界复制进行 [1,2,1]/4 卷积
    2) 经度方向周期循环进行相同卷积
    """
    # 纬度方向平滑
    pad_lat = np.pad(data, ((1,1),(0,0)), mode='edge')
    lat_s = (pad_lat[:-2,:] + 2*pad_lat[1:-1,:] + pad_lat[2:,:]) * 0.25

    # 经度方向平滑
    pad_lon = np.concatenate((lat_s[:,-1:], lat_s, lat_s[:,:1]), axis=1)
    smooth = (pad_lon[:,:-2] + 2*pad_lon[:,1:-1] + pad_lon[:,2:]) * 0.25

    return smooth

def plot_contour(ax, height, lon, lat, subtitle):
    """
    在 Lambert Conformal 投影上，只画黑色等值线，
    样式参考示例图，无填色和 colorbar。
    """
    h_cyc, lon_cyc = add_cyclic_point(height, coord=lon)

    ax.set_extent([60, 180, 20, 80], crs=ccrs.PlateCarree())
    ax.coastlines('50m', linewidth=1)

    # 网格线
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linestyle='--', linewidth=0.5, color='gray')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = plt.FixedLocator(np.arange(60, 181, 30))
    gl.ylocator = plt.FixedLocator(np.arange(20, 81, 10))
    gl.xlabel_style = gl.ylabel_style = {'size': 8}

    # 等值线
    levels = np.arange(4800, 6000, 120)
    cs = ax.contour(lon_cyc, lat, h_cyc, levels=levels,
                    colors='black', linewidths=1.2,
                    transform=ccrs.PlateCarree())
    ax.clabel(cs, fmt='%d', inline=True, fontsize=8)

    # 左对齐小标题
    ax.set_title(subtitle, fontsize=10, loc='left')

if __name__ == '__main__':
    # 文件路径
    nc_path = r'D:\zhuomian\shuzhi\shiyan8\500hpa.nc'
    assert os.path.exists(nc_path), f"文件不存在：{nc_path}"

    # 中文字体设置
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 加载并预处理数据
    height, lon, lat = load_500hpa_height(nc_path)

    # 2. 快速九点平滑
    height_s = smooth_9point(height)

    # 3. 绘图
    fig = plt.figure(figsize=(10, 4))
    proj = ccrs.LambertConformal(
        central_longitude=120,
        central_latitude=50,
        standard_parallels=(30, 60)
    )
    ax1 = fig.add_subplot(1, 2, 1, projection=proj)
    plot_contour(ax1, height, lon, lat, '(a) 原始数据')

    ax2 = fig.add_subplot(1, 2, 2, projection=proj)
    plot_contour(ax2, height_s, lon, lat, '(b) 九点平滑后的结果')

    # 总标题
    fig.suptitle('1979年1月10日 500 hPa 重力位势高度场', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
