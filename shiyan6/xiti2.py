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

# —————————————— 初始化场和动能数组 ——————————————
# 初始化数组
u = np.zeros((nlon, ntime))
# 设置初始条件
u[:, 0] = np.sin(np.arange(1, nlon+1) * np.pi * dx)
# 初始化动能数组
uk = np.zeros(ntime)
uk[0] = np.sum(u[:, 0]**2 / 2)

# —————————————— 数值积分主循环 ——————————————
for step_id in range(1, ntime):
    # 计算内部点
    for lon_id in range(1, nlon-1):
        # Lax-Wendroff格式
        u[lon_id, step_id] = u[lon_id, step_id-1] - \
            0.5 * c * dt / dx * (u[lon_id+1, step_id-1] - u[lon_id-1, step_id-1]) + \
            (c * dt / dx)**2 * (u[lon_id+1, step_id-1] - 2*u[lon_id, step_id-1] + u[lon_id-1, step_id-1])
        uk[step_id] += u[lon_id, step_id]**2 / 2
    
    # 边界条件保持不变
    u[nlon-1, step_id] = u[nlon-1, step_id-1]
    u[0, step_id] = u[0, step_id-1]
    
    # 添加边界点的动能贡献
    uk[step_id] += u[nlon-1, step_id]**2 / 2 + u[0, step_id]**2 / 2

# —————————————— 绘图 ——————————————
plt.figure(figsize=(10, 6))
plt.plot(range(1, ntime+1), uk, '--k', linewidth=1.5)
plt.xlabel('积分步数', fontsize=14)
plt.ylabel('动能', fontsize=14)
plt.title('Lax-Wendroff格式', fontsize=14)
plt.grid(True)
plt.show()