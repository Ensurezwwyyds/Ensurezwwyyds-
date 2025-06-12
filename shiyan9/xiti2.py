import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from tqdm import tqdm

def load_200hpa_fields(nc_file):
    """
    读取200hPa的u、v、z要素场数据的连续三个时次
    返回：三个时次的要素场数据、经度、纬度
    """
    print("正在读取数据...")
    with xr.open_dataset(nc_file, engine='netcdf4', decode_times=False) as ds:
        # 选择200hPa层的连续三个时次数据
        fields = {}
        for var in ['u', 'v', 'z']:
            data = ds[var].sel(pressure_level=200).isel(valid_time=slice(0, 3))
            if var == 'z':
                data = data / 9.80665  # 转换为位势米
            fields[var] = data.values

        # 获取原始经纬度数据
        lon = ds.longitude.values
        lat = ds.latitude.values

        # 确保经度范围在[-180, 180)之间
        lon = ((lon + 180) % 360) - 180
        idx = np.argsort(lon)
        lon = lon[idx]
        
        # 调整所有时次数据的经度顺序
        for var in fields:
            fields[var] = fields[var][:, :, idx]

    return fields, lon, lat

def time_smooth(data_n_minus_1, data_n, data_n_plus_1, s=0.5):
    """
    对中间时次进行三点时间平滑
    参数：
    data_n_minus_1, data_n, data_n_plus_1: 连续三个时次的数据
    s: 平滑系数，默认0.5
    """
    data_smooth = data_n.copy()
    data_smooth[1:-1, 1:-1] = data_n[1:-1, 1:-1] + s * (
        data_n_minus_1[1:-1, 1:-1] + data_n_plus_1[1:-1, 1:-1] - 2.0 * data_n[1:-1, 1:-1]
    ) / 2.0
    return data_smooth

def plot_fields(field, field_smooth, lon, lat, var_name, units):
    """
    在两个子图中分别绘制原始场和平滑后的场
    """
    # 添加循环点
    field_cyc, lon_cyc = add_cyclic_point(field, coord=lon)
    field_smooth_cyc, _ = add_cyclic_point(field_smooth, coord=lon)

    # 创建兰勃特投影
    proj = ccrs.LambertConformal(
        central_longitude=120,
        central_latitude=50,
        standard_parallels=(30, 60)
    )

    # 创建画布和子图
    fig = plt.figure(figsize=(15, 7))

    # 绘制原始场
    ax1 = fig.add_subplot(121, projection=proj)
    plot_single_field(ax1, field_cyc, lon_cyc, lat, f"原始{var_name}场")

    # 绘制平滑场
    ax2 = fig.add_subplot(122, projection=proj)
    plot_single_field(ax2, field_smooth_cyc, lon_cyc, lat, f"时间平滑后{var_name}场")

    # 添加总标题
    title = f'1979年1月10日 200hPa {var_name}场分析\n'
    subtitle = f'单位：{units}'
    plt.suptitle(title + subtitle, fontsize=16, y=0.98)
    
    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.2, top=0.85)

def plot_single_field(ax, field, lon, lat, title):
    """
    绘制单个要素场子图
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
    gl.xlabel_style = gl.ylabel_style = {'size': 10}

    # 绘制等值线
    cs = ax.contour(lon, lat, field, colors='black', linewidths=1.2,
                    transform=ccrs.PlateCarree())
    
    # 添加等值线标签
    plt.clabel(cs, fmt='%d', inline=True, fontsize=9)

    # 添加子图标题
    ax.set_title(title, fontsize=14, pad=10, y=1.05)

def main():
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 数据文件路径
    nc_file = 'geo_197901_200hpa.nc'
    if not os.path.exists(nc_file):
        raise FileNotFoundError(f"找不到文件：{nc_file}")

    # 读取数据
    fields, lon, lat = load_200hpa_fields(nc_file)

    # 变量名称和单位
    var_info = {
        'u': ('纬向风', 'm/s'),
        'v': ('经向风', 'm/s'),
        'z': ('位势高度', 'm')
    }

    # 对每个要素场进行时间平滑并绘图
    print("正在进行时间平滑和绘图...")
    for var in tqdm(fields):
        # 获取三个时次的数据
        data_n_minus_1 = fields[var][0]
        data_n = fields[var][1]
        data_n_plus_1 = fields[var][2]

        # 进行时间平滑
        data_smooth = time_smooth(data_n_minus_1, data_n, data_n_plus_1)

        # 绘制对比图
        plot_fields(data_n, data_smooth, lon, lat, var_info[var][0], var_info[var][1])
        plt.show()

if __name__ == '__main__':
    main()