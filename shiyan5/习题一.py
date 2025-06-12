import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

def time_integration_backward_central(u0, c, dx, dt, nsteps):
    """
    时间层后差 + 空间层中央差分格式积分一维线性平流方程
    ∂u/∂t + c ∂u/∂x = 0
    边界条件：固定边界 u[0]=u[0], u[-1]=u[-1]
    """
    nlon = u0.size
    u = np.zeros((nlon, nsteps))
    uk = np.zeros(nsteps)
    
    # 初始条件
    u[:, 0] = u0
    uk[0] = np.sum(u[:, 0]**2) / 2.0
    
    # 时间积分
    for n in range(1, nsteps):
        # 后差时间 + 中心差分空间
        for i in range(1, nlon - 1):
            u[i, n] = u[i, n-1] - 0.5 * c * dt / dx * (u[i+1, n] - u[i-1, n])
        
        # 固定边界条件
        u[0, n] = u[0, n-1]
        u[-1, n] = u[-1, n-1]
        
        # 累计动能
        uk[n] = np.sum(u[:, n]**2) / 2.0
    
    return u, uk

def time_integration_backward_forward(u0, c, dx, dt, nsteps):
    """
    时间层后差 + 空间层前差分格式积分一维线性平流方程
    ∂u/∂t + c ∂u/∂x = 0
    边界条件：固定边界 u[0]=u[0], u[-1]=u[-1]
    """
    nlon = u0.size
    u = np.zeros((nlon, nsteps))
    uk = np.zeros(nsteps)
    
    # 初始条件
    u[:, 0] = u0
    uk[0] = np.sum(u[:, 0]**2) / 2.0
    
    # 时间积分
    for n in range(1, nsteps):
        # 后差时间 + 前差分空间
        for i in range(1, nlon - 1):
            u[i, n] = u[i, n-1] - c * dt / dx * (u[i+1, n] - u[i, n])
        
        # 固定边界条件
        u[0, n] = u[0, n-1]
        u[-1, n] = u[-1, n-1]
        
        # 累计动能
        uk[n] = np.sum(u[:, n]**2) / 2.0
    
    return u, uk

def main():
    # 参数设置
    nlon = 20        # 空间格点数
    ntime = 3000     # 时间步数
    dx = 0.05        # 空间步长
    dt = 0.004       # 时间步长
    c = 1.5          # 平流速度
    
    # 初始场 u(x,0) = sin(pi x)，x = i*dx, i=1..nlon
    x = np.arange(1, nlon+1) * dx
    u0 = np.sin(np.pi * x)
    
    # 时间积分
    u1, uk1 = time_integration_backward_central(u0, c, dx, dt, ntime)
    u2, uk2 = time_integration_backward_forward(u0, c, dx, dt, ntime)
    
    # 创建图形窗口
    plt.figure(figsize=(16, 5))
    
    # 绘制中央差分结果（只显示前35步）
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(35), uk1[:35], '--k', linewidth=1.5)
    plt.ylim(0, 200)
    plt.xlabel('积分步数', fontsize=14)
    plt.ylabel(r'动能 $\sum_i u_i^2/2$', fontsize=14)
    plt.title('时间层后差，空间层中央差分', fontsize=16)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    # 绘制前差分结果
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(ntime), uk2, '--k', linewidth=1.5)
    plt.ylim(0, 10)
    plt.xlabel('积分步数', fontsize=14)
    plt.ylabel(r'动能 $\sum_i u_i^2/2$', fontsize=14)
    plt.title('时间层后差，空间层前差分', fontsize=16)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    # 调整布局并显示
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()