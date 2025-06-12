"""
平衡初值积分正压原始方程模式
改进版：通过优化的Helmholtz分解和更精确的平衡条件求解流函数场
基于：风场 = 无辐散风场 + 有辐散风场
其中无辐散风场由流函数ψ确定：u = -∂ψ/∂y, v = ∂ψ/∂x
"""

import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

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
    print("  正在进行网格插值...")
    Lon, Lat = np.meshgrid(lon, lat)
    pts_src = np.column_stack((Lon.ravel(), Lat.ravel()))
    pts_tgt = np.column_stack((lmda.ravel(), phai.ravel()))

    # 使用进度条显示插值过程
    with tqdm(total=3, desc="插值进度") as pbar:
        ui_lin = griddata(pts_src, u.ravel(), pts_tgt, method='linear')
        pbar.update(1)
        vi_lin = griddata(pts_src, v.ravel(), pts_tgt, method='linear')
        pbar.update(1)
        zi_lin = griddata(pts_src, z.ravel(), pts_tgt, method='linear')
        pbar.update(1)
    
    # 最近邻插值填补缺失值
    ui_nn  = griddata(pts_src, u.ravel(), pts_tgt, method='nearest')
    vi_nn  = griddata(pts_src, v.ravel(), pts_tgt, method='nearest')
    zi_nn  = griddata(pts_src, z.ravel(), pts_tgt, method='nearest')

    ui = ui_lin.copy(); vi = vi_lin.copy(); zi = zi_lin.copy()
    mask = np.isnan(ui); ui[mask] = ui_nn[mask]
    mask = np.isnan(vi); vi[mask] = vi_nn[mask]
    mask = np.isnan(zi); zi[mask] = zi_nn[mask]

    return ui.reshape(m,n), vi.reshape(m,n), zi.reshape(m,n)

def compute_vorticity_improved(u, v, d):
    """改进的涡度计算 ζ = ∂v/∂x - ∂u/∂y"""
    m, n = u.shape
    vorticity = np.zeros_like(u)
    
    # 使用四阶精度的中心差分
    for i in range(2, m-2):
        for j in range(2, n-2):
            # 四阶中心差分：f'(x) ≈ [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)]/(12h)
            dvdx = (-v[i, j+2] + 8*v[i, j+1] - 8*v[i, j-1] + v[i, j-2]) / (12*d)
            dudy = (-u[i+2, j] + 8*u[i+1, j] - 8*u[i-1, j] + u[i-2, j]) / (12*d)
            vorticity[i, j] = dvdx - dudy
    
    # 边界使用二阶差分
    # 内部边界（距离边界1个格点）
    for i in [1, m-2]:
        for j in range(1, n-1):
            dvdx = (v[i, j+1] - v[i, j-1]) / (2*d)
            dudy = (u[i+1, j] - u[i-1, j]) / (2*d) if i == 1 else (u[i-1, j] - u[i+1, j]) / (2*d)
            vorticity[i, j] = dvdx - dudy
    
    for j in [1, n-2]:
        for i in range(1, m-1):
            dvdx = (v[i, j+1] - v[i, j-1]) / (2*d) if j == 1 else (v[i, j-1] - v[i, j+1]) / (2*d)
            dudy = (u[i+1, j] - u[i-1, j]) / (2*d)
            vorticity[i, j] = dvdx - dudy
    
    return vorticity

def compute_divergence_improved(u, v, d):
    """改进的散度计算 δ = ∂u/∂x + ∂v/∂y"""
    m, n = u.shape
    divergence = np.zeros_like(u)
    
    # 使用四阶精度的中心差分
    for i in range(2, m-2):
        for j in range(2, n-2):
            dudx = (-u[i, j+2] + 8*u[i, j+1] - 8*u[i, j-1] + u[i, j-2]) / (12*d)
            dvdy = (-v[i+2, j] + 8*v[i+1, j] - 8*v[i-1, j] + v[i-2, j]) / (12*d)
            divergence[i, j] = dudx + dvdy
    
    # 边界使用二阶差分
    for i in [1, m-2]:
        for j in range(1, n-1):
            dudx = (u[i, j+1] - u[i, j-1]) / (2*d)
            dvdy = (v[i+1, j] - v[i-1, j]) / (2*d) if i == 1 else (v[i-1, j] - v[i+1, j]) / (2*d)
            divergence[i, j] = dudx + dvdy
    
    for j in [1, n-2]:
        for i in range(1, m-1):
            dudx = (u[i, j+1] - u[i, j-1]) / (2*d) if j == 1 else (u[i, j-1] - u[i, j+1]) / (2*d)
            dvdy = (v[i+1, j] - v[i-1, j]) / (2*d)
            divergence[i, j] = dudx + dvdy
    
    return divergence

def solve_poisson_improved(rhs, d, max_iter=500, tolerance=1e-5):
    """改进的泊松方程求解器，使用SOR迭代法"""
    m, n = rhs.shape
    psi = np.zeros((m, n))
    omega = 1.8  # SOR松弛因子
    
    print(f"  使用SOR方法求解泊松方程，最大迭代次数: {max_iter}")
    
    with tqdm(total=max_iter, desc="SOR迭代") as pbar:
        for iteration in range(max_iter):
            psi_old = psi.copy()
            
            # SOR迭代
            for i in range(1, m-1):
                for j in range(1, n-1):
                    residual = (psi[i+1, j] + psi[i-1, j] + psi[i, j+1] + psi[i, j-1] 
                               - 4*psi[i, j] - rhs[i, j]*d**2)
                    psi[i, j] += omega * residual / 4
            
            # 检查收敛性
            if iteration % 50 == 0:
                error = np.max(np.abs(psi - psi_old))
                if error < tolerance:
                    print(f"\n  SOR迭代在第{iteration}次收敛，误差: {error:.2e}")
                    break
            
            pbar.update(1)
    
    return psi

def helmholtz_decomposition_improved(u, v, d):
    """
    改进的Helmholtz分解：将风场分解为无辐散和有辐散部分
    使用更精确的数值方法和边界处理
    """
    print("  计算涡度场...")
    vorticity = compute_vorticity_improved(u, v, d)
    
    print("  计算散度场...")
    divergence = compute_divergence_improved(u, v, d)
    
    print("  求解流函数...")
    psi = solve_poisson_improved(vorticity, d)
    
    print("  求解速度势...")
    chi = solve_poisson_improved(divergence, d)
    
    # 计算无辐散风场（使用改进的差分格式）
    m, n = u.shape
    u_psi = np.zeros_like(u)
    v_psi = np.zeros_like(v)
    
    # 使用四阶差分计算梯度
    for i in range(2, m-2):
        for j in range(2, n-2):
            # u_ψ = -∂ψ/∂y
            u_psi[i, j] = -(-psi[i+2, j] + 8*psi[i+1, j] - 8*psi[i-1, j] + psi[i-2, j]) / (12*d)
            # v_ψ = ∂ψ/∂x
            v_psi[i, j] = (-psi[i, j+2] + 8*psi[i, j+1] - 8*psi[i, j-1] + psi[i, j-2]) / (12*d)
    
    # 边界使用二阶差分
    for i in [1, m-2]:
        for j in range(1, n-1):
            u_psi[i, j] = -(psi[i+1, j] - psi[i-1, j]) / (2*d) if i == 1 else -(psi[i-1, j] - psi[i+1, j]) / (2*d)
            v_psi[i, j] = (psi[i, j+1] - psi[i, j-1]) / (2*d)
    
    for j in [1, n-2]:
        for i in range(1, m-1):
            u_psi[i, j] = -(psi[i+1, j] - psi[i-1, j]) / (2*d)
            v_psi[i, j] = (psi[i, j+1] - psi[i, j-1]) / (2*d) if j == 1 else (psi[i, j-1] - psi[i, j+1]) / (2*d)
    
    return u_psi, v_psi, psi, chi

def geostrophic_balance_adjustment(u_psi, v_psi, phi, f, d, alpha=0.15):
    """
    地转平衡调整：根据重力位势场对无辐散风场进行调整
    使其更好地满足地转平衡关系
    """
    print("  计算地转风场...")
    g = 9.8
    m, n = u_psi.shape
    u_geo = np.zeros_like(u_psi)
    v_geo = np.zeros_like(v_psi)
    
    # 使用改进的差分格式计算地转风
    for i in range(2, m-2):
        for j in range(2, n-2):
            if np.abs(f[i, j]) > 1e-10:
                # u_g = -(1/f) * ∂Φ/∂y
                dphidy = (-phi[i+2, j] + 8*phi[i+1, j] - 8*phi[i-1, j] + phi[i-2, j]) / (12*d)
                u_geo[i, j] = -dphidy / f[i, j]
                
                # v_g = (1/f) * ∂Φ/∂x
                dphidx = (-phi[i, j+2] + 8*phi[i, j+1] - 8*phi[i, j-1] + phi[i, j-2]) / (12*d)
                v_geo[i, j] = dphidx / f[i, j]
            else:
                u_geo[i, j] = u_psi[i, j]
                v_geo[i, j] = v_psi[i, j]
    
    # 边界处理
    for i in [1, m-2]:
        for j in range(1, n-1):
            if np.abs(f[i, j]) > 1e-10:
                dphidy = (phi[i+1, j] - phi[i-1, j]) / (2*d) if i == 1 else (phi[i-1, j] - phi[i+1, j]) / (2*d)
                u_geo[i, j] = -dphidy / f[i, j]
                dphidx = (phi[i, j+1] - phi[i, j-1]) / (2*d)
                v_geo[i, j] = dphidx / f[i, j]
            else:
                u_geo[i, j] = u_psi[i, j]
                v_geo[i, j] = v_psi[i, j]
    
    # 自适应权重调整
    print("  进行自适应平衡调整...")
    u_balanced = np.zeros_like(u_psi)
    v_balanced = np.zeros_like(v_psi)
    
    for i in range(m):
        for j in range(n):
            # 根据科里奥利参数调整权重
            local_alpha = alpha * (1 + np.abs(f[i, j]) / (2*7.292e-5))
            local_alpha = min(local_alpha, 0.5)  # 限制最大权重
            
            u_balanced[i, j] = (1-local_alpha) * u_psi[i, j] + local_alpha * u_geo[i, j]
            v_balanced[i, j] = (1-local_alpha) * v_psi[i, j] + local_alpha * v_geo[i, j]
    
    return u_balanced, v_balanced

def advanced_balance_initialization(u, v, phi, f, d):
    """
    高级平衡初值化方法
    1. 改进的Helmholtz分解
    2. 地转平衡调整
    3. 多尺度平滑处理
    4. 能量守恒检查
    """
    print("开始高级平衡初值化...")
    
    # 第一步：Helmholtz分解
    print("第一步：Helmholtz分解")
    u_psi, v_psi, psi, chi = helmholtz_decomposition_improved(u, v, d)
    
    # 第二步：地转平衡调整
    print("第二步：地转平衡调整")
    u_balanced, v_balanced = geostrophic_balance_adjustment(u_psi, v_psi, phi, f, d)
    
    # 第三步：多尺度平滑
    print("第三步：多尺度平滑处理")
    # 大尺度平滑
    u_smooth1 = gaussian_filter(u_balanced, sigma=2.0)
    v_smooth1 = gaussian_filter(v_balanced, sigma=2.0)
    
    # 小尺度平滑
    u_smooth2 = gaussian_filter(u_balanced, sigma=0.5)
    v_smooth2 = gaussian_filter(v_balanced, sigma=0.5)
    
    # 组合不同尺度
    u_final = 0.7 * u_smooth2 + 0.3 * u_smooth1
    v_final = 0.7 * v_smooth2 + 0.3 * v_smooth1
    
    # 第四步：能量守恒检查和调整
    print("第四步：能量守恒检查")
    energy_orig = np.mean(u**2 + v**2)
    energy_final = np.mean(u_final**2 + v_final**2)
    energy_ratio = np.sqrt(energy_orig / energy_final)
    
    # 调整以保持能量
    u_final *= energy_ratio
    v_final *= energy_ratio
    
    print(f"  能量调整比例: {energy_ratio:.4f}")
    
    return u_final, v_final, psi

def main():
    """主函数"""
    print("="*60)
    print("高级平衡初值积分正压原始方程模式")
    print("="*60)
    
    # 参数设置
    m, n = 41, 17
    d = 300000.0  # 网格距离 (m)
    clat, clon = 45.0, 120.0  # 中心纬度和经度
    
    print(f"网格设置: {m}×{n}, 网格距离: {d/1000:.0f} km")
    print(f"投影中心: ({clat}°N, {clon}°E)")
    
    print("\n读取ERA5数据...")
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
                z_data = ds['gh'].values * 9.80665
            elif 'hgt' in ds.variables:
                z_data = ds['hgt'].values * 9.80665
            else:
                print("警告：未找到高度场变量，将基于风场估算")
                z_data = None
            
            lon = ds['longitude'].values if 'longitude' in ds.variables else ds['lon'].values
            lat = ds['latitude'].values if 'latitude' in ds.variables else ds['lat'].values
            
            # 处理数据维度
            print(f"原始数据形状 - u: {u_data.shape}, v: {v_data.shape}")
            
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
            
            # 如果没有高度场数据，基于风场估算位势场
            if z_data is None:
                print("基于地转平衡关系估算位势场...")
                omega_earth = 7.292e-5
                Lon_orig, Lat_orig = np.meshgrid(lon, lat)
                f_coriolis_orig = 2 * omega_earth * np.sin(np.deg2rad(Lat_orig))
                f_coriolis_orig = np.where(np.abs(f_coriolis_orig) < 1e-10, 1e-10, f_coriolis_orig)
                
                # g = 9.80665
                z = np.zeros_like(u) # 初始化位势场 Phi
                
                # 改进的位势场估算 (积分方法)
                # 注意：这种积分方法对路径敏感，可能不精确
                dlat_rad = np.deg2rad(np.abs(lat[1] - lat[0])) if len(lat) > 1 else 0
                dlon_rad = np.deg2rad(np.abs(lon[1] - lon[0])) if len(lon) > 1 else 0
                R_earth = 6.371e6
                
                # 从赤道附近开始积分可能更稳定，或使用参考点
                # 这里简单地从一个边界开始
                if len(lat) > 1:
                    for i in range(1, len(lat)):
                        # dPhi = -f*u*dy  => Phi_i = Phi_{i-1} - f_avg * u_avg * dy
                        # dy = R_earth * dlat_rad
                        # 使用中点值进行积分可能更好
                        f_avg = (f_coriolis_orig[i, :] + f_coriolis_orig[i-1, :]) / 2
                        u_avg = (u[i, :] + u[i-1, :]) / 2
                        z[i, :] = z[i-1, :] - f_avg * u_avg * (R_earth * dlat_rad)
                
                if len(lon) > 1:
                    for j in range(1, len(lon)):
                        # dPhi = f*v*dx => Phi_j = Phi_{j-1} + f_avg * v_avg * dx
                        # dx = R_earth * cos(lat) * dlon_rad
                        f_avg = (f_coriolis_orig[:, j] + f_coriolis_orig[:, j-1]) / 2
                        v_avg = (v[:, j] + v[:, j-1]) / 2
                        cos_lat_avg = np.cos(np.deg2rad((Lat_orig[:, j] + Lat_orig[:, j-1]) / 2))
                        z[:, j] = z[:, j-1] + f_avg * v_avg * (R_earth * cos_lat_avg * dlon_rad)
                
                # 添加基础位势值和平滑处理
                z = z + 50000 * 9.80665 # 假设500hPa高度约5000m，转换为位势
                z = gaussian_filter(z, sigma=1.5) # 增加平滑强度
                
        else:
            raise KeyError("geo_197901.nc文件中未找到u和v变量")
        
    except Exception as e:
        print(f"读取数据时出错: {e}")
        print("使用改进的模拟数据")
        # 创建更真实的模拟数据
        lon_sim = np.arange(0, 360, 2.5) # 减少模拟数据点数以加快测试
        lat_sim = np.arange(-90, 90.25, 2.5)
        lon = lon_sim
        lat = lat_sim
        Lon, Lat_grid = np.meshgrid(lon, lat)
        
        # 创建具有多尺度特征的风场
        u = (25 * np.sin(np.deg2rad(2*Lon)) * np.cos(np.deg2rad(Lat_grid)) + 
             15 * np.cos(np.deg2rad(Lon/2)) * np.sin(np.deg2rad(Lat_grid)) +
             8 * np.sin(np.deg2rad(3*Lon)) * np.cos(np.deg2rad(2*Lat_grid)))
        v = (20 * np.cos(np.deg2rad(Lon)) * np.sin(np.deg2rad(2*Lat_grid)) + 
             12 * np.sin(np.deg2rad(Lon/3)) * np.cos(np.deg2rad(Lat_grid)) +
             6 * np.cos(np.deg2rad(2*Lon)) * np.sin(np.deg2rad(3*Lat_grid)))
        # z 是位势 Phi (m^2/s^2)
        z = (50000 * 9.80665 + 4000 * 9.80665 * np.sin(np.deg2rad(Lon)) * np.cos(np.deg2rad(Lat_grid)) + 
             3000 * 9.80665 * np.cos(np.deg2rad(2*Lon)) * np.sin(np.deg2rad(Lat_grid)) +
             1500 * 9.80665 * np.sin(np.deg2rad(Lon/2)) * np.cos(np.deg2rad(3*Lat_grid)))
    
    print(f"\n数据统计:")
    print(f"  u风场: [{np.nanmin(u):.2f}, {np.nanmax(u):.2f}] m/s")
    print(f"  v风场: [{np.nanmin(v):.2f}, {np.nanmax(v):.2f}] m/s")
    print(f"  位势场: [{np.nanmin(z):.0f}, {np.nanmax(z):.0f}] m²/s²")
    
    print("\n进行坐标变换...")
    # 兰伯特投影坐标变换
    rm, f_proj, lmda_degree, phai_degree = cmf(d, clat, clon, m, n)
    
    print("\n插值到投影网格...")
    # 插值到投影网格
    ua, va, za = interp_proj_grid(u, v, z, lmda_degree, phai_degree, m, n, lon, lat)
    
    print(f"\n投影后数据统计:")
    print(f"  u风场: [{np.nanmin(ua):.2f}, {np.nanmax(ua):.2f}] m/s")
    print(f"  v风场: [{np.nanmin(va):.2f}, {np.nanmax(va):.2f}] m/s")
    print(f"  位势场: [{np.nanmin(za):.0f}, {np.nanmax(za):.0f}] m²/s²")
    
    print("\n计算平衡风场...")
    # 使用高级平衡初值化方法
    # 注意：advanced_balance_initialization 需要投影后的科氏参数 f_proj
    up, vp, psi = advanced_balance_initialization(ua, va, za, f_proj, d)
    
    # 在绘图前检查数据
    print("\n绘图前数据检查:")
    for name, data_array in zip(["ua", "va", "za", "up", "vp", "psi"], [ua, va, za, up, vp, psi]):
        if data_array is not None:
            print(f"  {name}: min={np.nanmin(data_array):.2f}, max={np.nanmax(data_array):.2f}, has_nan={np.isnan(data_array).any()}, has_inf={np.isinf(data_array).any()}")
        else:
            print(f"  {name}: is None")

    print("\n绘制对比图...")
    # 创建改进的对比图
    fig = plt.figure(figsize=(18, 15))
    
    # 支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建子图
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)
    
    # 原始u风场
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.LambertConformal(
        central_longitude=clon, central_latitude=clat))
    u_min, u_max = np.nanmin(ua), np.nanmax(ua)
    levels_u = np.linspace(u_min, u_max, 15) if u_max - u_min > 1 else np.arange(-30, 31, 5)
    cs1 = ax1.contour(lmda_degree, phai_degree, ua, levels=levels_u, 
                     colors='blue', linewidths=1.2, transform=ccrs.PlateCarree())
    if hasattr(cs1, 'levels') and len(cs1.levels) > 0: 
        ax1.clabel(cs1, inline=True, fontsize=9, fmt='%.1f')
    ax1.coastlines(linewidth=1, color='gray')
    gl1 = ax1.gridlines(draw_labels=True, alpha=0.5)
    gl1.top_labels = False; gl1.right_labels = False
    gl1.xlabel_style = {'size': 10}; gl1.ylabel_style = {'size': 10}
    ax1.set_title('(a) 原始u风场 (m/s)', fontsize=12, fontweight='bold')
    
    # 平衡u风场
    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.LambertConformal(
        central_longitude=clon, central_latitude=clat))
    up_min, up_max = np.nanmin(up), np.nanmax(up)
    levels_up = np.linspace(up_min, up_max, 15) if up_max - up_min > 1 else levels_u
    cs2 = ax2.contour(lmda_degree, phai_degree, up, levels=levels_up, 
                     colors='red', linewidths=1.2, transform=ccrs.PlateCarree())
    if hasattr(cs2, 'levels') and len(cs2.levels) > 0:
        ax2.clabel(cs2, inline=True, fontsize=9, fmt='%.1f')
    ax2.coastlines(linewidth=1, color='gray')
    gl2 = ax2.gridlines(draw_labels=True, alpha=0.5)
    gl2.top_labels = False; gl2.right_labels = False
    gl2.xlabel_style = {'size': 10}; gl2.ylabel_style = {'size': 10}
    ax2.set_title('(b) 平衡u风场 (m/s)', fontsize=12, fontweight='bold')
    
    # 原始v风场
    ax3 = fig.add_subplot(gs[1, 0], projection=ccrs.LambertConformal(
        central_longitude=clon, central_latitude=clat))
    v_min, v_max = np.nanmin(va), np.nanmax(va)
    levels_v = np.linspace(v_min, v_max, 15) if v_max - v_min > 1 else np.arange(-25, 26, 5)
    cs3 = ax3.contour(lmda_degree, phai_degree, va, levels=levels_v, 
                     colors='blue', linewidths=1.2, transform=ccrs.PlateCarree())
    if hasattr(cs3, 'levels') and len(cs3.levels) > 0:
        ax3.clabel(cs3, inline=True, fontsize=9, fmt='%.1f')
    ax3.coastlines(linewidth=1, color='gray')
    gl3 = ax3.gridlines(draw_labels=True, alpha=0.5)
    gl3.top_labels = False; gl3.right_labels = False
    gl3.xlabel_style = {'size': 10}; gl3.ylabel_style = {'size': 10}
    ax3.set_title('(c) 原始v风场 (m/s)', fontsize=12, fontweight='bold')
    
    # 平衡v风场
    ax4 = fig.add_subplot(gs[1, 1], projection=ccrs.LambertConformal(
        central_longitude=clon, central_latitude=clat))
    vp_min, vp_max = np.nanmin(vp), np.nanmax(vp)
    levels_vp = np.linspace(vp_min, vp_max, 15) if vp_max - vp_min > 1 else levels_v
    cs4 = ax4.contour(lmda_degree, phai_degree, vp, levels=levels_vp, 
                     colors='red', linewidths=1.2, transform=ccrs.PlateCarree())
    if hasattr(cs4, 'levels') and len(cs4.levels) > 0:
        ax4.clabel(cs4, inline=True, fontsize=9, fmt='%.1f')
    ax4.coastlines(linewidth=1, color='gray')
    gl4 = ax4.gridlines(draw_labels=True, alpha=0.5)
    gl4.top_labels = False; gl4.right_labels = False
    gl4.xlabel_style = {'size': 10}; gl4.ylabel_style = {'size': 10}
    ax4.set_title('(d) 平衡v风场 (m/s)', fontsize=12, fontweight='bold')

    # 流函数 psi
    ax5 = fig.add_subplot(gs[2, 0], projection=ccrs.LambertConformal(
        central_longitude=clon, central_latitude=clat))
    psi_min, psi_max = np.nanmin(psi), np.nanmax(psi)
    levels_psi = np.linspace(psi_min, psi_max, 20) if psi_max - psi_min > 1e-6 else (np.array([psi_min]) if not np.isnan(psi_min) else 10)
    cs5 = ax5.contour(lmda_degree, phai_degree, psi, levels=levels_psi, 
                     colors='green', linewidths=1.2, transform=ccrs.PlateCarree())
    if hasattr(cs5, 'levels') and len(cs5.levels) > 0:
        ax5.clabel(cs5, inline=True, fontsize=9, fmt='%.0e')
    ax5.coastlines(linewidth=1, color='gray')
    gl5 = ax5.gridlines(draw_labels=True, alpha=0.5)
    gl5.top_labels = False
    gl5.right_labels = False
    gl5.xlabel_style = {'size': 10}
    gl5.ylabel_style = {'size': 10}
    ax5.set_title('(e) 流函数 $\psi$ ($\mathrm{m}^2/\mathrm{s}$)', fontsize=12, fontweight='bold') # 使用LaTeX显示单位

    # 统计信息
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    stats_text = f"原始风场范围:\n  u: [{np.nanmin(ua):.2f}, {np.nanmax(ua):.2f}] m/s\n  v: [{np.nanmin(va):.2f}, {np.nanmax(va):.2f}] m/s\n"
    stats_text += f"平衡风场范围:\n  u: [{np.nanmin(up):.2f}, {np.nanmax(up):.2f}] m/s\n  v: [{np.nanmin(vp):.2f}, {np.nanmax(vp):.2f}] m/s\n"
    
    # 计算统计量 (使用 n-1 作为分母，如果需要严格的n-1，需要自定义函数或调整np.mean)
    # np.mean 默认使用 n 作为分母。对于RMSE，通常是N。
    # 对于样本标准差，numpy.std默认ddof=0 (n)，若要n-1，则ddof=1。
    # 这里RMSE的定义是均方根误差，所以分母是N。
    rmse_u = np.sqrt(np.mean((up - ua)**2))
    rmse_v = np.sqrt(np.mean((vp - va)**2))
    
    # 相关系数 np.corrcoef 内部处理了均值和标准差的计算
    # 确保输入是一维数组
    flat_ua = ua.flatten()
    flat_up = up.flatten()
    flat_va = va.flatten()
    flat_vp = vp.flatten()
    
    # 移除NaN值以计算相关系数
    valid_indices_u = ~np.isnan(flat_ua) & ~np.isnan(flat_up)
    valid_indices_v = ~np.isnan(flat_va) & ~np.isnan(flat_vp)

    corr_u = np.corrcoef(flat_ua[valid_indices_u], flat_up[valid_indices_u])[0, 1] if np.sum(valid_indices_u) > 1 else np.nan
    corr_v = np.corrcoef(flat_va[valid_indices_v], flat_vp[valid_indices_v])[0, 1] if np.sum(valid_indices_v) > 1 else np.nan
    
    energy_orig = np.nanmean(ua**2 + va**2)
    energy_bal = np.nanmean(up**2 + vp**2)
    energy_ratio = np.sqrt(energy_bal / energy_orig) if energy_orig > 1e-9 else 0

    stats_text += f"\n统计指标:\n  RMSE u: {rmse_u:.2f} m/s\n  RMSE v: {rmse_v:.2f} m/s\n"
    stats_text += f"  Corr u: {corr_u:.2f}\n  Corr v: {corr_v:.2f}\n"
    stats_text += f"  能量保持率: {energy_ratio:.2f}"
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    ax6.set_title('(f) 统计信息', fontsize=12, fontweight='bold')

    fig.suptitle('原始风场与平衡风场对比 (高级平衡初值化)', fontsize=16, fontweight='bold')
    plt.subplots_adjust(top=0.92)
    plt.show()
    
    print("\n统计指标:")
    print(f"  RMSE u: {rmse_u:.2f} m/s, RMSE v: {rmse_v:.2f} m/s")
    print(f"  相关系数 u: {corr_u:.2f}, 相关系数 v: {corr_v:.2f}")
    print(f"  能量保持率: {energy_ratio:.2f}")

    # 返回主要数据以供外部使用或测试
    return ua, va, up, vp, psi, za, lmda_degree, phai_degree

if __name__ == "__main__":
    # 解包时需要对应 main 函数返回的变量数量
    main_output = main()
    if main_output is not None:
        ua_main, va_main, up_main, vp_main, psi_main, za_main, lmda_main, phai_main = main_output
        print("\nMain function executed successfully and returned data.")
    else:
        print("\nMain function did not return data (possibly due to an error before plotting or no return statement).")