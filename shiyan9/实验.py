import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from tqdm import tqdm

def time_smooth_fields(u, v, z, smooth_factor=0.5):
    """
    时间平滑子程序
    参数：
        u, v, z: 三个时间层的风场和位势高度场数据，每个数组形状为(3, lat, lon)
        smooth_factor: 平滑系数，默认为0.5
    返回：
        平滑后的u、v、z场
    """
    def smooth_single_field(field):
        # 获取中间时刻的场
        field_smooth = field[1].copy()
        # 对非边界点进行平滑
        field_smooth[1:-1, 1:-1] = field[1][1:-1, 1:-1] + smooth_factor * (
            field[0][1:-1, 1:-1] + field[2][1:-1, 1:-1] - 2.0 * field[1][1:-1, 1:-1]
        ) / 2.0
        return field_smooth

    # 分别对u、v、z场进行平滑
    u_smooth = smooth_single_field(u)
    v_smooth = smooth_single_field(v)
    z_smooth = smooth_single_field(z)

    return u_smooth, v_smooth, z_smooth

def load_500hpa_data(nc_file):
    """
    读取500hPa位势高度场数据
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
        
        # 调整数据的经度顺序
        height = height[:, :, idx]

    return height, lon, lat

def plot_height_fields(height_original, height_smooth, lon, lat):
    """
    绘制原始场和平滑后的场对比图
    """
    # 添加循环点
    height_original_cyc, lon_cyc = add_cyclic_point(height_original, coord=lon)
    height_smooth_cyc, _ = add_cyclic_point(height_smooth, coord=lon)

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
    plot_single_field(ax1, height_original_cyc, lon_cyc, lat, "原始500hPa位势高度场")

    # 绘制平滑场
    ax2 = fig.add_subplot(122, projection=proj)
    plot_single_field(ax2, height_smooth_cyc, lon_cyc, lat, "时间平滑后500hPa位势高度场")

    # 添加总标题
    title = '1979年1月10日 500hPa位势高度场分析\n'
    subtitle = '单位：m'
    plt.suptitle(title + subtitle, fontsize=16, y=0.98)
    
    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.2, top=0.85)

def plot_single_field(ax, field, lon, lat, title):
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
    gl.xlabel_style = gl.ylabel_style = {'size': 10}

    # 绘制等值线
    levels = np.arange(5000, 5751, 150)
    cs = ax.contour(lon, lat, field, levels=levels,
                    colors='black', linewidths=1.2,
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
    nc_file = 'geo_197901.nc'
    if not os.path.exists(nc_file):
        raise FileNotFoundError(f"找不到文件：{nc_file}")

    # 读取数据
    height, lon, lat = load_500hpa_data(nc_file)

    # 进行时间平滑
    print("正在进行时间平滑...")
    # 由于示例中只需要对z场进行平滑，我们传入相同的数据作为u和v
    _, _, height_smooth = time_smooth_fields(height, height, height)

    # 绘制对比图
    print("正在绘制高度场对比图...")
    plot_height_fields(height[1], height_smooth, lon, lat)

    # 显示图像
    plt.show()

if __name__ == '__main__':
    main()