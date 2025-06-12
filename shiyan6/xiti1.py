import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# —————————————— 参数设置 ——————————————
nlon = 20        # 空间格点数
ntime = 300      # 时间步数
dx = 0.05        # 空间步长
dt = 0.004       # 时间步长
c = 1.5          # 平流速度

# 无量纲参数
lambda_ = 0.5 * c * dt / dx    # 用于中央差分的系数
r = c * dt / dx                # 用于前差格式的系数

# —————————————— 初始化场和动能数组 ——————————————
# 初始条件 u(i,0) = sin(i * pi * dx)
x = np.arange(1, nlon+1) * dx
u_central = np.zeros((nlon, ntime))
u_forward = np.zeros((nlon, ntime))
u_central[:, 0] = np.sin(x * np.pi)
u_forward[:, 0] = u_central[:, 0].copy()

# 动能：uk[n] = sum(u[:,n]**2 / 2)
uk_central = np.zeros(ntime)
uk_forward = np.zeros(ntime)
uk_central[0] = np.sum(u_central[:, 0]**2) / 2
uk_forward[0] = uk_central[0]

# —————————————— 数值积分主循环 ——————————————
for n in range(1, ntime):
    # —— 时间层后差，空间层中央差 —— 
    u_old = u_central[:, n-1]
    u_new = np.zeros(nlon)
    # 边界不变
    u_new[0] = u_old[0]
    u_new[-1] = u_old[-1]
    # 内部点
    for i in range(1, nlon-1):
        u_new[i] = u_old[i] - lambda_ * (u_old[i+1] - u_old[i-1])
    u_central[:, n] = u_new
    uk_central[n] = np.sum(u_new**2) / 2

    # —— 时间层后差，空间层前差 —— 
    u_old = u_forward[:, n-1]
    u_new = np.zeros(nlon)
    # 边界不变
    u_new[0] = u_old[0]
    u_new[-1] = u_old[-1]
    # 内部点
    for i in range(1, nlon-1):
        u_new[i] = u_old[i] - r * (u_old[i+1] - u_old[i])
    u_forward[:, n] = u_new
    uk_forward[n] = np.sum(u_new**2) / 2

# —————————————— 绘图 ——————————————
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(ntime), uk_central, '--k', linewidth=1.5)
plt.title("时间层后差，空间层中央差")
plt.xlabel("积分步数")
plt.ylabel("动能 $\\sum u^2/2$")
plt.ylim(0, 10)

plt.subplot(1, 2, 2)
plt.plot(range(40), uk_forward[:40], '--k', linewidth=1.5)  # 只绘制前40步
plt.title("时间层后差，空间层前差")
plt.xlabel("积分步数")
plt.ylabel("动能 $\\sum u^2/2$")
plt.ylim(0, 100)

plt.tight_layout()
plt.show()