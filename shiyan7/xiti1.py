import numpy as np
import matplotlib.pyplot as plt

# —————— 中文与负号支持 ——————
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def central_dirichlet(u, c, dx, dt):
    """
    Dirichlet 边界下的 central (leap-frog) 格式积分一维线性平流方程。
    u: ndarray，shape=(nlon, ntime)，u[:,0] 已给初值
    返回：u 完整场，uk 每步总动能
    对应 time_integration.m 中 case 'central' 的实现。:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
    """
    nlon, ntime = u.shape
    β = c * dt / dx

    uk = np.zeros(ntime)
    uk[0] = np.sum(u[:,0]**2 / 2)

    # 启动步（step_id=2）
    for i in range(1, nlon-1):
        u[i,1] = u[i,0] - 0.5 * β * (u[i+1,0] - u[i-1,0])
    # 边界固定
    u[0,1]   = u[0,0]
    u[-1,1]  = u[-1,0]
    uk[1]    = np.sum(u[:,1]**2 / 2)

    # 正式 leap-frog 步（step_id=3:ntime）
    for n in range(2, ntime):
        # interior
        u[1:-1,n] = u[1:-1,n-2] - β * (u[2:,n-1] - u[:-2,n-1])
        # 边界固定
        u[0,n]    = u[0,n-1]
        u[-1,n]   = u[-1,n-1]
        uk[n]     = np.sum(u[:,n]**2 / 2)

    return u, uk

if __name__ == "__main__":
    # — 参数设置，与 xiti1.m 中相同，只改 c 和初始场 :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
    nlon, ntime = 20, 3000
    dx, dt, c   = 0.05, 0.004, 2.5

    β = c * dt / dx
    print(f"Courant 数 β = {β:.3f} （|β|≤1 → 稳定）")

    # 初始化
    x = np.arange(nlon) * dx
    u = np.zeros((nlon, ntime))
    u[:,0] = np.sin(np.pi * x) + 1.5

    # 积分 & 计算动能
    u, uk = central_dirichlet(u, c, dx, dt)

    # 绘图
    plt.figure(figsize=(8,4))
    plt.plot(np.arange(ntime), uk, '--k', linewidth=1.5)
    plt.xlim(0, ntime)
    plt.ylim(0, 100)
    plt.xlabel('积分步数', fontsize=14)
    plt.ylabel('动能', fontsize=14)
    plt.title('纬向总动能随积分次数的变化', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
