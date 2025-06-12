import numpy as np
import matplotlib.pyplot as plt

# —————————————— 支持中文显示 ——————————————
plt.rcParams['font.sans-serif'] = ['SimHei']      # 用于显示中文标签
plt.rcParams['axes.unicode_minus'] = False        # 用于正常显示负号

def time_integration_central(u, c, dx, dt):
    """
    中心差分格式积分一维线性平流方程：
      (u_i^{n+1} - u_i^{n-1})/(2dt) + c*(u_{i+1}^n - u_{i-1}^n)/(2dx) = 0

    首步用显式半步中心差分，后续用中心时差 + 中心空差。
    
    参数
    ----
    u  : ndarray, shape (nlon, ntime)
         储存积分结果，u[:,0] 已给定初值
    c  : 浮点数，流速
    dx : 浮点数，空间步长
    dt : 浮点数，时间步长

    返回
    ----
    u  : ndarray, shape (nlon, ntime) ，积分后的解
    uk : ndarray, shape (ntime,)      ，各步动能
    """
    nlon, ntime = u.shape
    uk = np.zeros(ntime)
    # 初始动能
    uk[0] = np.sum(u[:,0]**2 / 2)

    beta = c * dt / dx

    # ——— 第一步（MATLAB 中 step_id=2） ———
    for i in range(1, nlon-1):
        u[i,1] = u[i,0] - 0.5 * beta * (u[i+1,0] - u[i-1,0])
    # 两端固定边界
    u[0,1]   = u[0,0]
    u[-1,1]  = u[-1,0]
    uk[1]    = np.sum(u[:,1]**2 / 2)

    # ——— 后续步（MATLAB 中 step_id=3:ntime） ———
    for n in range(2, ntime):
        for i in range(1, nlon-1):
            u[i,n] = u[i,n-2] - beta * (u[i+1,n-1] - u[i-1,n-1])
        # 边界固定
        u[0,n]   = u[0,n-1]
        u[-1,n]  = u[-1,n-1]
        uk[n]    = np.sum(u[:,n]**2 / 2)

    return u, uk

def main():
    # ——— 参数设置 ———
    nlon  = 20
    ntime = 3000
    dx    = 0.05
    dt    = 0.004
    c     = 1.5

    # 初始化 u，满足 u(i,0) = sin(pi * x_i)，其中 x_i = i * dx
    u = np.zeros((nlon, ntime))
    x = np.arange(1, nlon+1) * dx
    u[:,0] = np.sin(np.pi * x)

    # 调用积分函数
    u, uk = time_integration_central(u, c, dx, dt)

    # ——— 绘图 ———
    plt.figure(figsize=(8,4))
    plt.plot(np.arange(1, ntime+1), uk, '--k', linewidth=1.5)
    plt.ylim(0, 20)
    plt.xlabel('积分步数', fontsize=14)
    plt.ylabel('动能', fontsize=14)
    plt.title('纬向总动能随积分次数的变化', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
