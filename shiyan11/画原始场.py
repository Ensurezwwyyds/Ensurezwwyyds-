import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def load_and_plot_era5_data(filename):
    """
    读取ERA5数据并绘制指定区域的原始位势高度场（墨卡托投影）
    """
    # 设置matplotlib支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    try:
        print(f"正在读取ERA5数据文件: {filename}")
        ds = xr.open_dataset(filename)
        print("数据文件读取成功！")
        
        # 自动识别变量
        z_var = None
        for var in ds.data_vars:
            if 'geopotential' in var.lower() or 'z' == var.lower():
                z_var = var
                break
        
        # 识别坐标变量
        lon_var = None
        lat_var = None
        for coord in ds.coords:
            if 'lon' in coord.lower():
                lon_var = coord
            elif 'lat' in coord.lower():
                lat_var = coord
        
        print(f"识别的变量: 位势高度={z_var}, 经度={lon_var}, 纬度={lat_var}")
        
        # 提取数据
        if z_var:
            z500 = ds[z_var]
            print(f"原始数据形状: {z500.shape}")
            print(f"数据维度: {z500.dims}")
            
            # 处理多维数据 - 选择第一个时间步和第一个层次
            if z500.ndim > 2:
                # 如果有时间维度，选择第一个时间步
                if 'time' in z500.dims:
                    z500 = z500.isel(time=0)
                    print("选择第一个时间步")
                
                # 如果有层次维度，选择第一个层次
                if len(z500.dims) > 2:
                    # 找到非经纬度的维度
                    other_dims = [dim for dim in z500.dims if dim not in ['latitude', 'longitude', 'lat', 'lon']]
                    if other_dims:
                        z500 = z500.isel({other_dims[0]: 0})
                        print(f"选择第一个{other_dims[0]}层次")
                
                # 确保最终是2D数据
                z500 = z500.squeeze()
                print(f"处理后数据形状: {z500.shape}")
            
            # 转换为numpy数组
            z500 = z500.values
            
            # 如果是位势，转换为位势高度
            if z500.max() > 100000:
                z500 = z500 / 9.8
                print("已转换位势为位势高度")
        else:
            raise ValueError("未找到位势高度变量")
        
        # 提取坐标
        lon = ds[lon_var].values
        lat = ds[lat_var].values
        
        # 定义积分区域范围（根据图像确定）
        lon_min, lon_max = 60, 180
        lat_min, lat_max = 12, 60
        
        # 选择区域数据
        lon_mask = (lon >= lon_min) & (lon <= lon_max)
        lat_mask = (lat >= lat_min) & (lat <= lat_max)
        
        lon_region = lon[lon_mask]
        lat_region = lat[lat_mask]
        
        # 获取区域索引
        lon_indices = np.where(lon_mask)[0]
        lat_indices = np.where(lat_mask)[0]
        
        # 提取区域数据
        z500_region = z500[np.ix_(lat_indices, lon_indices)]
        
        print(f"区域数据形状: z500={z500_region.shape}, lon={len(lon_region)}, lat={len(lat_region)}")
        print(f"经度范围: {lon_region.min():.1f}° - {lon_region.max():.1f}°")
        print(f"纬度范围: {lat_region.min():.1f}° - {lat_region.max():.1f}°")
        print(f"位势高度范围: {z500_region.min():.1f} - {z500_region.max():.1f} m")
        
        # 创建网格
        LAT_region, LON_region = np.meshgrid(lat_region, lon_region, indexing='ij')
        
        # 使用墨卡托投影绘制
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 墨卡托投影等值线图
        ax1 = plt.subplot(2, 2, 1, projection=ccrs.Mercator())
        ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # 添加地理特征
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax1.add_feature(cfeature.LAND, alpha=0.3, color='lightgray')
        ax1.add_feature(cfeature.OCEAN, alpha=0.3, color='lightblue')
        
        # 绘制等值线
        levels = np.arange(int(z500_region.min()/50)*50, int(z500_region.max()/50)*50+50, 50)
        cs1 = ax1.contour(LON_region, LAT_region, z500_region, levels=levels, 
                         colors='black', linewidths=1, transform=ccrs.PlateCarree())
        ax1.clabel(cs1, inline=True, fontsize=8, fmt='%d')
        
        # 设置网格线和标签
        ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        ax1.set_title('ERA5 500hPa 位势高度场 - 墨卡托投影等值线图', fontsize=14)
        
        # 2. 墨卡托投影填色图
        ax2 = plt.subplot(2, 2, 2, projection=ccrs.Mercator())
        ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # 添加地理特征
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
        
        # 绘制填色图
        cs2 = ax2.contourf(LON_region, LAT_region, z500_region, levels=30, 
                          cmap='RdYlBu_r', transform=ccrs.PlateCarree())
        cbar2 = plt.colorbar(cs2, ax=ax2, shrink=0.8)
        cbar2.set_label('位势高度 (m)', fontsize=12)
        
        # 叠加等值线
        cs2_lines = ax2.contour(LON_region, LAT_region, z500_region, levels=levels[::2], 
                               colors='black', linewidths=0.5, transform=ccrs.PlateCarree())
        
        ax2.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        ax2.set_title('ERA5 500hPa 位势高度场 - 墨卡托投影填色图', fontsize=14)
        
        # 3. 传统经纬度投影对比图
        ax3 = plt.subplot(2, 2, 3)
        cs3 = ax3.contour(LON_region, LAT_region, z500_region, levels=levels, 
                         colors='blue', linewidths=1)
        ax3.clabel(cs3, inline=True, fontsize=8, fmt='%d')
        ax3.set_title('传统经纬度投影对比图', fontsize=14)
        ax3.set_xlabel('经度 (°)')
        ax3.set_ylabel('纬度 (°)')
        ax3.grid(True, alpha=0.3)
        
        # 4. 数据统计信息
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        stats_text = f"""
积分区域数据统计信息:

• 数据源: {filename}
• 积分区域: {lon_min}°E - {lon_max}°E, {lat_min}°N - {lat_max}°N
• 网格尺寸: {z500_region.shape[0]} × {z500_region.shape[1]}
• 位势高度范围: {z500_region.min():.1f} - {z500_region.max():.1f} m
• 平均值: {z500_region.mean():.1f} m
• 标准差: {z500_region.std():.1f} m
• 投影方式: 墨卡托投影
• 等值线间隔: 50 m
        """
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # 绘制大图 - 墨卡托投影详细图
        fig2 = plt.figure(figsize=(16, 10))
        ax_main = plt.axes(projection=ccrs.Mercator())
        ax_main.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # 添加地理特征
        ax_main.add_feature(cfeature.COASTLINE, linewidth=1)
        ax_main.add_feature(cfeature.BORDERS, linewidth=0.8)
        ax_main.add_feature(cfeature.LAND, alpha=0.2, color='lightgray')
        ax_main.add_feature(cfeature.OCEAN, alpha=0.2, color='lightblue')
        
        # 绘制填色图
        cs_main = ax_main.contourf(LON_region, LAT_region, z500_region, levels=40, 
                                  cmap='RdYlBu_r', transform=ccrs.PlateCarree(), alpha=0.8)
        
        # 绘制等值线
        cs_lines = ax_main.contour(LON_region, LAT_region, z500_region, levels=levels, 
                                  colors='black', linewidths=1, transform=ccrs.PlateCarree())
        ax_main.clabel(cs_lines, inline=True, fontsize=10, fmt='%d')
        
        # 添加颜色条
        cbar_main = plt.colorbar(cs_main, ax=ax_main, shrink=0.8, pad=0.05)
        cbar_main.set_label('位势高度 (m)', fontsize=14)
        
        # 设置网格线和标签
        gl = ax_main.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                              linewidth=0.5, color='gray', alpha=0.7)
        gl.top_labels = False
        gl.right_labels = False
        
        ax_main.set_title('ERA5 500hPa 位势高度场 - 积分区域墨卡托投影详细图', fontsize=16, pad=20)
        
        plt.show()
        
        return z500_region, lon_region, lat_region
        
    except Exception as e:
        print(f"读取数据时出错: {e}")
        return None, None, None

if __name__ == "__main__":
    print("ERA5积分区域位势高度场可视化程序（墨卡托投影）")
    print("=" * 50)
    
    # 读取并绘制数据
    z500, lon, lat = load_and_plot_era5_data('geo_197901.nc')
    
    if z500 is not None:
        print("\n积分区域原始场绘制完成！")
        print(f"区域数据包含 {z500.shape[0]} × {z500.shape[1]} 个网格点")
        print("使用墨卡托投影显示")
    else:
        print("无法读取数据文件")