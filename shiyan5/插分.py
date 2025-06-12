import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

def time_integration_forward(u0, c, dx, dt, nsteps):
    """
    显式前向差分（时间）+ 中心差分（空间）积分一维线性平流方程
    ∂u/∂t + c ∂u/∂x = 0
    边界条件：固定边界 u[0]=u[0], u[-1]=u[-1]
    
    输入：
      u0      -- 初始一维场，长度 nlon
      c       -- 平流速度常数
      dx      -- 空间步长
      dt      -- 时间步长
      nsteps  -- 积分步数
    返回：
      u       -- 始终保存所有时刻的 u，shape=(nlon, nsteps)
      uk      -- 每步的总动能 time series，shape=(nsteps,)
    """
    nlon = u0.size
    u = np.zeros((nlon, nsteps))
    uk = np.zeros(nsteps)
    
    # 初始条件
    u[:, 0] = u0
    uk[0] = np.sum(u[:, 0]**2) / 2.0
    
    # 时间积分
    for n in range(1, nsteps):
        # 前向 Euler 时间 + 中心差分空间
        for i in range(1, nlon - 1):
            u[i, n] = (u[i, n-1]
                       - 0.5 * c * dt / dx
                       * (u[i+1, n-1] - u[i-1, n-1]))
        # 固定边界条件
        u[0, n]  = u[0,   n-1]
        u[-1, n] = u[-1,  n-1]
        
        # 累计动能
        uk[n] = np.sum(u[:, n]**2) / 2.0
    
    return u, uk

def main():
    # 参数设置
    nlon  = 20       # 空间格点数
    ntime = 3000     # 时间步数
    dx    = 0.05     # 空间步长
    dt    = 0.004    # 时间步长
    c     = 1.5      # 平流速度
    
    # 初始场 u(x,0) = sin(pi x)，x = i*dx, i=1..nlon
    x  = np.arange(1, nlon+1) * dx
    u0 = np.sin(np.pi * x)
    
    # 时间积分
    u, uk = time_integration_forward(u0, c, dx, dt, ntime)
    
    # 绘图：动能随积分步数变化
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(ntime), uk, '--k', linewidth=1.5)
    plt.ylim(0, 200)
    
    # 坐标轴标签和标题，设置字体大小
    plt.xlabel('积分步数', fontsize=14)
    plt.ylabel(r'动能 $\sum_i u_i^2/2$', fontsize=14)
    plt.title('显式前向+中心差分方案动能演变', fontsize=16)
    
    # 刻度字体大小
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # 添加网格和加粗坐标轴线
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    plt.tight_layout()
    plt.show()  

if __name__ == '__main__':
    main()
