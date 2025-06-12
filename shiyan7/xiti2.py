import numpy as np
import matplotlib.pyplot as plt

# —————— 支持中文和负号 ——————
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ——— 通用参数 ———
nlon = 20
dx   = 0.1
dt   = 0.004
c    = 1.5

# 初始场：u(i,1) = sin(i·π·dx/2)
x  = np.arange(1, nlon+1) * dx/2 * np.pi
u0 = np.sin(x)

# -----------------------------------
# 格式一：5500 步，日志坐标
# -----------------------------------
ntime1 = 5500
u1 = np.zeros((nlon, ntime1))
u1[:, 0] = u0
u1[:, 1] = u0            # MATLAB: u(:,2)=u(:,1)
uk1 = np.zeros(ntime1)
uk1[0] = uk1[1] = np.sum(u0**2/2)

coef1 = dt/(4*dx)        # =0.25*dt/dx

for n in range(2, ntime1):
    # 内点
    for i in range(1, nlon-1):
        up = u1[i+1, n-1]
        um = u1[i-1, n-1]
        ui = u1[i  , n-1]
        u1[i, n] = (
            u1[i, n-2]
            - coef1 * ((up + ui)**2 - (ui + um)**2)
        )
    # 周期边界
    u1[0    , n] = u1[0    , n-1]
    u1[-1   , n] = u1[-1   , n-1]
    uk1[n] = np.sum(u1[:, n]**2/2)

# -----------------------------------
# 格式二：仅 3 步，线性放大
# -----------------------------------
ntime2 = 3
u2 = np.zeros((nlon, ntime2))
u2[:, 0] = u0
u2[:, 1] = u0
uk2 = np.zeros(ntime2)
uk2[0] = uk2[1] = np.sum(u0**2/2)

for n in range(2, ntime2):
    # 周期边界前先算平均场
    ubar = 0.5 * (u2[:, n-1] + u2[:, n-2])
    for i in range(nlon):
        ip = (i+1) % nlon
        im = (i-1) % nlon
        u2[i, n] = (
            u2[i, n-2]
            - dt/(6*dx)*(ubar[ip] + ubar[i] + ubar[im])*(ubar[ip] - ubar[im])
        )
    uk2[n] = np.sum(u2[:, n]**2/2)

# ——— 绘图 ———
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

# 左：格式一（对数纵轴）
ax1.plot(np.arange(1, ntime1+1), uk1, '--k', linewidth=1.5)
ax1.set_yscale('log')
ax1.set_xlim(1, ntime1)
ax1.set_ylim(1e-1, 1e5)
ax1.set_xlabel('积分步数')
ax1.set_ylabel('动能')
ax1.set_title('格式一：总动能（对数刻度）')
ax1.grid(True)

# 右：格式二（放大前 3 步）
ax2.plot(np.arange(1, ntime2+1), uk2, '--k', linewidth=1.5)
ax2.set_xlim(1, 3)
ax2.set_ylim(5.00, 5.00012)
ax2.set_xlabel('积分步数')
ax2.set_title('格式二：总动能（放大前三步）')
ax2.grid(True)

plt.tight_layout()
plt.show()
