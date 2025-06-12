import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from tqdm import tqdm

def load_500hpa_height_series(nc_file):
    """
    读取500hPa位势高度场数据的连续三个时次
    返回：三个时次的高度场数据、经度、纬度
    """
    print("正在读取数据...")
    with xr.open_dataset(nc_file, engine='netcdf4', decode_times=False) as ds:
        # 选择500hPa层的连续三个时次数据
        z = ds['z'].sel(pressure_level=500).isel(valid_time=slice(0, 3))
        height = (z / 9.80665).values  # 转换为米

        # 获取原始经纬度数据
        lon = ds.longitude.values
        lat = ds.latitude.values

        # 确保经度范围在[-180, 180)之间
        lon = ((lon + 180) % 360) - 180
        idx = np.argsort(lon)
        lon = lon[idx]
        
        # 调整所有时次数据的经度顺序
        height = height[:, :, idx]

    return height[0], height[1], height[2], lon, lat

def time_smooth(za, zb, zc, s=0.5):
    """
    对中间时次进行三点时间平滑
    参数：
    za, zb, zc: 连续三个时次的数据
    s: 平滑系数，默认0.5
    """
    zb_smooth = zb.copy()
    zb_smooth[1:-1, 1:-1] = zb[1:-1, 1:-1] + s * (
        za[1:-1, 1:-1] + zc[1:-1, 1:-1] - 2.0 * zb[1:-1, 1:-1]
    ) / 2.0
    return zb_smooth

def plot_height_fields(height, height_smooth, lon, lat):
    """
    在两个子图中分别绘制原始场和平滑后的场
    """
    # 添加循环点
    height_cyc, lon_cyc = add_cyclic_point(height, coord=lon)
    height_smooth_cyc, _ = add_cyclic_point(height_smooth, coord=lon)

    # 创建兰勃特投影
    proj = ccrs.LambertConformal(
        central_longitude=120,
        central_latitude=50,
        standard_parallels=(30, 60)
    )

    # 创建画布和子图
    fig = plt.figure(figsize=(15, 6))

    # 绘制原始场
    ax1 = fig.add_subplot(121, projection=proj)
    plot_single_field(ax1, height_cyc, lon_cyc, lat, "(a) 原始场")

    # 绘制平滑场
    ax2 = fig.add_subplot(122, projection=proj)
    plot_single_field(ax2, height_smooth_cyc, lon_cyc, lat, "(b) 时间平滑场")

    # 添加总标题
    plt.suptitle('1979年1月10日500hPa重力位势高度场', fontsize=14, y=1.02)

def plot_single_field(ax, height, lon, lat, title):
    """
    绘制单个高度场子图
    """
    # 设置地图范围
    ax.set_extent([60, 180, 20, 80], crs=ccrs.PlateCarree())
    
    # 添加海岸线
    ax.coastlines('50m', linewidth=0.8)

    # 添加网格线
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linestyle='--', linewidth=0.5, color='gray')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = plt.FixedLocator(np.arange(60, 181, 30))
    gl.ylocator = plt.FixedLocator(np.arange(20, 81, 10))
    gl.xlabel_style = gl.ylabel_style = {'size': 8}

    # 绘制等值线
    levels = np.arange(5000, 5751, 150)
    cs = ax.contour(lon, lat, height, levels=levels,
                    colors='black', linewidths=1.2,
                    transform=ccrs.PlateCarree())
    
    # 添加等值线标签
    plt.clabel(cs, fmt='%d', inline=True, fontsize=8)

    # 添加子图标题
    ax.set_title(title, fontsize=12, pad=10)

def main():
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 数据文件路径
    nc_file = 'geo_197901.nc'
    if not os.path.exists(nc_file):
        raise FileNotFoundError(f"找不到文件：{nc_file}")

    # 读取数据
    za, zb, zc, lon, lat = load_500hpa_height_series(nc_file)

    # 进行时间平滑
    print("正在进行时间平滑...")
    zb_smooth = time_smooth(za, zb, zc)

    # 绘制对比图
    print("正在绘制高度场对比图...")
    plot_height_fields(zb, zb_smooth, lon, lat)

    # 调整布局并显示
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()