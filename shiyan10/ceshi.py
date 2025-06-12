"""
平衡初值积分正压原始方程模式
改进版：通过Helmholtz分解和更合理的平衡条件求解流函数场
基于：风场 = 无辐散风场 + 有辐散风场
其中无辐散风场由流函数ψ确定：u = -∂ψ/∂y, v = ∂ψ/∂x
"""

import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from tqdm import tqdm

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

def compute_vorticity(u, v, d):
    """计算涡度 ζ = ∂v/∂x - ∂u/∂y"""
    vorticity = np.zeros_like(u)
    
    # 中心差分计算偏导数
    dvdx = np.zeros_like(v)
    dudy = np.zeros_like(u)
    
    dvdx[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2*d)
    dudy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2*d)
    
    vorticity = dvdx - dudy
    return vorticity

def compute_divergence(u, v, d):
    """计算散度 δ = ∂u/∂x + ∂v/∂y"""
    divergence = np.zeros_like(u)
    
    # 中心差分计算偏导数
    dudx = np.zeros_like(u)
    dvdy = np.zeros_like(v)
    
    dudx[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2*d)
    dvdy[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2*d)
    
    divergence = dudx + dvdy
    return divergence

def solve_poisson_equation(rhs, d, boundary_condition='neumann'):
    """求解泊松方程 ∇²ψ = rhs"""
    m, n = rhs.shape
    
    # 构建五点差分格式的系数矩阵
    N = m * n
    diagonals = []
    offsets = []
    
    # 主对角线
    main_diag = -4 * np.ones(N)
    diagonals.append(main_diag)
    offsets.append(0)
    
    # 上下对角线 (±1)
    upper_diag = np.ones(N-1)
    lower_diag = np.ones(N-1)
    
    # 处理边界条件
    for i in range(N-1):
        if (i+1) % n == 0:  # 行末尾
            upper_diag[i] = 0
    
    diagonals.extend([upper_diag, lower_diag])
    offsets.extend([1, -1])
    
    # 左右对角线 (±n)
    left_diag = np.ones(N-n)
    right_diag = np.ones(N-n)
    
    diagonals.extend([left_diag, right_diag])
    offsets.extend([n, -n])
    
    # 构建稀疏矩阵
    A = diags(diagonals, offsets, shape=(N, N), format='csr')
    
    # 处理边界条件
    if boundary_condition == 'dirichlet':
        # 边界点设为0
        for i in range(N):
            row, col = divmod(i, n)
            if row == 0 or row == m-1 or col == 0 or col == n-1:
                A[i, :] = 0
                A[i, i] = 1
                rhs.flat[i] = 0
    
    # 求解线性方程组
    rhs_flat = rhs.flatten() * d**2
    psi_flat = spsolve(A, rhs_flat)
    
    return psi_flat.reshape(m, n)

def helmholtz_decomposition(u, v, d):
    """
    Helmholtz分解：将风场分解为无辐散和有辐散部分
    u = u_ψ + u_χ, v = v_ψ + v_χ
    其中 u_ψ = -∂ψ/∂y, v_ψ = ∂ψ/∂x (无辐散)
         u_χ = ∂χ/∂x, v_χ = ∂χ/∂y (有辐散)
    """
    # 计算涡度和散度
    vorticity = compute_vorticity(u, v, d)
    divergence = compute_divergence(u, v, d)
    
    # 求解流函数：∇²ψ = ζ
    psi = solve_poisson_equation(vorticity, d)
    
    # 求解速度势：∇²χ = δ
    chi = solve_poisson_equation(divergence, d)
    
    # 计算无辐散风场
    u_psi = np.zeros_like(u)
    v_psi = np.zeros_like(v)
    
    # u_ψ = -∂ψ/∂y
    u_psi[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2*d)
    # v_ψ = ∂ψ/∂x
    v_psi[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2*d)
    
    # 计算有辐散风场
    u_chi = np.zeros_like(u)
    v_chi = np.zeros_like(v)
    
    # u_χ = ∂χ/∂x
    u_chi[1:-1, :] = (chi[2:, :] - chi[:-2, :]) / (2*d)
    # v_χ = ∂χ/∂y
    v_chi[:, 1:-1] = (chi[:, 2:] - chi[:, :-2]) / (2*d)
    
    return u_psi, v_psi, u_chi, v_chi, psi, chi

def balance_adjustment(u_psi, v_psi, phi, f, d, alpha=0.1):
    """
    平衡调整：根据重力位势场对无辐散风场进行调整
    使其更好地满足地转平衡关系
    """
    # 计算地转风
    g = 9.8
    u_geo = np.zeros_like(u_psi)
    v_geo = np.zeros_like(v_psi)
    
    # u_g = -(1/f) * ∂Φ/∂y
    u_geo[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2*d*f[:, 1:-1])
    # v_g = (1/f) * ∂Φ/∂x
    v_geo[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2*d*f[1:-1, :])
    
    # 避免除零
    mask = np.abs(f) < 1e-10
    u_geo[mask] = u_psi[mask]
    v_geo[mask] = v_psi[mask]
    
    # 平衡调整：在无辐散风场和地转风之间进行加权
    u_balanced = (1-alpha) * u_psi + alpha * u_geo
    v_balanced = (1-alpha) * v_psi + alpha * v_geo
    
    return u_balanced, v_balanced

def improved_balance_initialization(u, v, phi, f, d):
    """
    改进的平衡初值化方法
    1. 首先进行Helmholtz分解得到无辐散风场
    2. 然后根据重力位势场进行平衡调整
    3. 保持原始风场的主要特征
    """
    print("进行Helmholtz分解...")
    u_psi, v_psi, u_chi, v_chi, psi, chi = helmholtz_decomposition(u, v, d)
    
    print("进行平衡调整...")
    u_balanced, v_balanced = balance_adjustment(u_psi, v_psi, phi, f, d, alpha=0.2)
    
    # 应用平滑滤波减少数值噪声
    sigma = 1.0
    u_balanced = gaussian_filter(u_balanced, sigma=sigma)
    v_balanced = gaussian_filter(v_balanced, sigma=sigma)
    
    return u_balanced, v_balanced, psi

def main():
    """主函数"""
    # 参数设置
    m, n = 41, 17
    d = 300000.0  # 网格距离 (m)
    clat, clon = 45.0, 120.0  # 中心纬度和经度
    
    print("读取ERA5数据...")
    # 读取ERA5数据
    try:
        ds = xr.open_dataset('geo_197901.nc')
        print(f"数据集变量: {list(ds.variables.keys())}")
        
        # 读取变量
        if 'u' in ds.variables and 'v' in ds.variables:
            u_data = ds['u'].values
            v_data = ds['v'].values
            
            # 检查是否有z变量
            if 'z' in ds.variables:
                z_data = ds['z'].values
            elif 'gh' in ds.variables:
                z_data = ds['gh'].values * 9.8
            elif 'hgt' in ds.variables:
                z_data = ds['hgt'].values * 9.8
            else:
                print("警告：未找到高度场变量，使用模拟数据")
                z_data = None
            
            lon = ds['longitude'].values if 'longitude' in ds.variables else ds['lon'].values
            lat = ds['latitude'].values if 'latitude' in ds.variables else ds['lat'].values
            
            # 处理数据维度
            print(f"u数据形状: {u_data.shape}")
            
            # 选择时间步和层次
            if len(u_data.shape) == 4:
                nt_f = 0
                level_idx = 0
                u = u_data[nt_f, level_idx, :, :]
                v = v_data[nt_f, level_idx, :, :]
                if z_data is not None:
                    z = z_data[nt_f, level_idx, :, :]
            elif len(u_data.shape) == 3:
                if 'time' in ds.dims:
                    nt_f = 0
                    u = u_data[nt_f, :, :]
                    v = v_data[nt_f, :, :]
                    if z_data is not None:
                        z = z_data[nt_f, :, :]
                else:
                    level_idx = 0
                    u = u_data[level_idx, :, :]
                    v = v_data[level_idx, :, :]
                    if z_data is not None:
                        z = z_data[level_idx, :, :]
            else:
                u = u_data
                v = v_data
                if z_data is not None:
                    z = z_data
            
            # 如果没有高度场数据，创建一个合理的位势场
            if z_data is None:
                print("创建合理的位势场...")
                omega = 7.292e-5
                Lon, Lat = np.meshgrid(lon, lat)
                f_coriolis = 2 * omega * np.sin(np.deg2rad(Lat))
                f_coriolis = np.where(np.abs(f_coriolis) < 1e-10, 1e-10, f_coriolis)
                
                # 基于地转平衡关系估算位势场
                g = 9.8
                z = np.zeros_like(u)
                
                # 使用简化的地转关系积分
                dlat = np.deg2rad(lat[1] - lat[0])
                dlon = np.deg2rad(lon[1] - lon[0])
                R_earth = 6.371e6
                
                # 从风场积分得到位势场
                for i in range(1, len(lat)):
                    z[i, :] = z[i-1, :] - f_coriolis[i, :] * u[i, :] * dlat * R_earth
                
                for j in range(1, len(lon)):
                    z[:, j] = z[:, j-1] + f_coriolis[:, j] * v[:, j] * dlon * R_earth * np.cos(np.deg2rad(Lat[:, j]))
                
                z = z + 50000  # 添加基础位势值
                
        else:
            raise KeyError("geo_197901.nc文件中未找到u和v变量")
        
    except Exception as e:
        print(f"读取数据时出错: {e}")
        print("使用模拟数据")
        # 创建更真实的模拟数据
        lon = np.arange(0, 360, 0.25)
        lat = np.arange(-90, 90.25, 0.25)
        Lon, Lat = np.meshgrid(lon, lat)
        
        # 创建具有真实特征的风场
        u = (20 * np.sin(np.deg2rad(2*Lon)) * np.cos(np.deg2rad(Lat)) + 
             10 * np.cos(np.deg2rad(Lon/2)) * np.sin(np.deg2rad(Lat)))
        v = (15 * np.cos(np.deg2rad(Lon)) * np.sin(np.deg2rad(2*Lat)) + 
             8 * np.sin(np.deg2rad(Lon/3)) * np.cos(np.deg2rad(Lat)))
        z = (50000 + 3000 * np.sin(np.deg2rad(Lon)) * np.cos(np.deg2rad(Lat)) + 
             2000 * np.cos(np.deg2rad(2*Lon)) * np.sin(np.deg2rad(Lat)))
    
    print(f"数据范围 - u: [{u.min():.2f}, {u.max():.2f}], v: [{v.min():.2f}, {v.max():.2f}], z: [{z.min():.0f}, {z.max():.0f}]")
    
    print("进行坐标变换...")
    # 兰伯特投影坐标变换
    rm, f, lmda_degree, phai_degree = cmf(d, clat, clon, m, n)
    
    print("插值到投影网格...")
    # 插值到投影网格
    ua, va, za = interp_proj_grid(u, v, z, lmda_degree, phai_degree, m, n, lon, lat)
    
    print("计算平衡风场...")
    # 使用改进的平衡初值化方法
    up, vp, psi = improved_balance_initialization(ua, va, za, f, d)
    
    print("绘制对比图...")
    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), 
                            subplot_kw={'projection': ccrs.LambertConformal(
                                central_longitude=clon, central_latitude=clat)})
    
    # 原始u风场
    ax = axes[0, 0]
    cs = ax.contour(lmda_degree, phai_degree, ua, levels=np.arange(-20, 21, 4), 
                   colors='black', transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=8)
    ax.coastlines(linewidth=1, color='gray')
    gl = ax.gridlines(draw_labels=True)
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    ax.set_title('(a) 原始u风场', fontsize=12)
    
    # 平衡u风场
    ax = axes[0, 1]
    cs = ax.contour(lmda_degree, phai_degree, up, levels=np.arange(-20, 21, 4), 
                   colors='black', transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=8)
    ax.coastlines(linewidth=1, color='gray')
    gl = ax.gridlines(draw_labels=True)
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    ax.set_title('(b) 平衡u风场', fontsize=12)
    
    # 原始v风场
    ax = axes[1, 0]
    cs = ax.contour(lmda_degree, phai_degree, va, levels=np.arange(-20, 21, 4), 
                   colors='black', transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=8)
    ax.coastlines(linewidth=1, color='gray')
    gl = ax.gridlines(draw_labels=True)
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    ax.set_title('(c) 原始v风场', fontsize=12)
    
    # 平衡v风场
    ax = axes[1, 1]
    cs = ax.contour(lmda_degree, phai_degree, vp, levels=np.arange(-20, 21, 4), 
                   colors='black', transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=8)
    ax.coastlines(linewidth=1, color='gray')
    gl = ax.gridlines(draw_labels=True)
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    ax.set_title('(d) 平衡v风场', fontsize=12)
    
    # 使用subplots_adjust代替tight_layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, 
                       wspace=0.15, hspace=0.25)
    plt.suptitle('改进的平衡初值积分正压原始方程模式 - 风场对比', fontsize=14)
    plt.show()
    
    # 输出统计信息
    print("\n统计信息：")
    print(f"原始u风场范围: [{ua.min():.2f}, {ua.max():.2f}] m/s")
    print(f"平衡u风场范围: [{up.min():.2f}, {up.max():.2f}] m/s")
    print(f"原始v风场范围: [{va.min():.2f}, {va.max():.2f}] m/s")
    print(f"平衡v风场范围: [{vp.min():.2f}, {vp.max():.2f}] m/s")
    
    # 计算均方根误差
    rmse_u = np.sqrt(np.mean((ua - up)**2))
    rmse_v = np.sqrt(np.mean((va - vp)**2))
    print(f"\nu风场RMSE: {rmse_u:.3f} m/s")
    print(f"v风场RMSE: {rmse_v:.3f} m/s")
    
    # 计算相关系数
    corr_u = np.corrcoef(ua.flatten(), up.flatten())[0, 1]
    corr_v = np.corrcoef(va.flatten(), vp.flatten())[0, 1]
    print(f"u风场相关系数: {corr_u:.3f}")
    print(f"v风场相关系数: {corr_v:.3f}")
    
    # 计算能量保持率
    energy_orig = np.mean(ua**2 + va**2)
    energy_balanced = np.mean(up**2 + vp**2)
    energy_ratio = energy_balanced / energy_orig
    print(f"能量保持率: {energy_ratio:.3f}")
    
    return ua, va, up, vp, psi, za

if __name__ == "__main__":
    ua, va, up, vp, psi, za = main()