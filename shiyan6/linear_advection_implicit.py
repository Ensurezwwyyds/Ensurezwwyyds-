import numpy as np
import matplotlib.pyplot as plt

# 让 Matplotlib 正确显示中文和负号
plt.rcParams["font.sans-serif"] = ["SimHei"]      # 显示中文
plt.rcParams["axes.unicode_minus"] = False        # 正确显示负号

def time_integration(u, c, dx, dt, difference_scheme="backward"):
    """
    用欧拉格式（显式）或欧拉后插 / Lax-Wendroff 格式积分一维线性平流方程,
    并计算每步累计动能 uk。

    参数
    ----
    u : 2-D ndarray, shape (nlon, ntime)
        储存积分结果，u[:,0] 必须已填入初值。
    c : float
        平流速度
    dx : float
        空间分辨率
    dt : float
        时间步长
    difference_scheme : {'forward','backward'}
        'forward' – 前向欧拉 + 空间中心差分  
        'backward' – 欧拉后插 / Lax-Wendroff (与您给出的 MATLAB 公式一致)

    返回
    ----
    u  : ndarray  同输入，积分后的场
    uk : 1-D ndarray, shape (ntime,)
        每一步全场动能的累加值
    """
    nlon, ntime = u.shape
    uk = np.zeros(ntime)
    uk[0] = 0.5 * np.sum(u[:, 0] ** 2)

    if difference_scheme.lower() == "forward":
        for step in range(1, ntime):
            for i in range(1, nlon - 1):
                u[i, step] = (u[i, step - 1]
                              - 0.5 * c * dt / dx
                              * (u[i + 1, step - 1] - u[i - 1, step - 1]))
                uk[step] += 0.5 * u[i, step] ** 2

            u[0, step] = u[0, step - 1]
            u[-1, step] = u[-1, step - 1]
            uk[step] += 0.5 * (u[0, step] ** 2 + u[-1, step] ** 2)

    elif difference_scheme.lower() == "backward":
        coef1 = 0.5 * c * dt / dx
        coef2 = coef1 ** 2
        for step in range(1, ntime):
            for i in range(2, nlon - 2):
                u[i, step] = (u[i, step - 1]
                              - coef1 * (u[i + 1, step - 1] - u[i - 1, step - 1])
                              + coef2 * (u[i + 2, step - 1]
                                         - 2 * u[i, step - 1]
                                         + u[i - 2, step - 1]))
                uk[step] += 0.5 * u[i, step] ** 2

            for k in range(2):
                u[k, step] = u[k, step - 1]
                u[-(k + 1), step] = u[-(k + 1), step - 1]
                uk[step] += 0.5 * (u[k, step] ** 2 + u[-(k + 1), step] ** 2)
    else:
        raise ValueError("difference_scheme must be 'forward' or 'backward'")

    return u, uk

# -------------------- 主程序 --------------------
if __name__ == "__main__":
    nlon = 20
    dx   = 0.05
    c    = 1.5

    # 初值：u(x,0) = sin(pi x)，其中 x = (1:nlon) * dx
    u0 = np.sin(np.arange(1, nlon + 1) * np.pi * dx)

    # -------- 实验 1 : dt = 0.004，共 3000 步积分 ----------
    dt1    = 0.004
    ntime1 = 3000
    u1     = np.zeros((nlon, ntime1))
    u1[:, 0] = u0
    _, uk1 = time_integration(u1, c, dx, dt1, "backward")

    # -------- 实验 2 : dt = 0.04，仅 40 步积分 ----------
    dt2    = 0.04
    ntime2 = 40
    u2     = np.zeros((nlon, ntime2))
    u2[:, 0] = u0
    _, uk2 = time_integration(u2, c, dx, dt2, "backward")

    # ---------------- 绘图 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    axes[0].plot(np.arange(ntime1), uk1, "--k", linewidth=1.5)
    axes[0].set_ylim(-1, 100)
    axes[0].set_xlabel("积分步数", fontsize=12)
    axes[0].set_ylabel("动能",   fontsize=12)
    axes[0].text(-0.05, 1.05, "(a) dt = 0.004, 3000 步", transform=axes[0].transAxes)

    axes[1].plot(np.arange(ntime2), uk2, "--k", linewidth=1.5)
    axes[1].set_ylim(-1, 100)
    axes[1].set_xlabel("积分步数", fontsize=12)
    axes[1].set_ylabel("动能",   fontsize=12)
    axes[1].text(-0.05, 1.05, "(b) dt = 0.04, 40 步", transform=axes[1].transAxes)

    plt.show()
