# 习题一.py
import numpy as np
import matplotlib.pyplot as plt

# ———— 中文字体配置 ————
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False            # 负号正常显示

# ———— 参数设置 ————
S  = 0.5    # 平滑系数
dx = 1      # 网格间距
dy = dx     # 有时 dy 可不同，这里两者相同
L  = np.arange(1, 37)  # 波长范围 1,2,…,36

# ———— 计算响应函数 ————
# 先把角度转换为弧度：180./L * dx 是度
argx = np.deg2rad(180.0 / L * dx)
argy = np.deg2rad(180.0 / L * dy)

# 五点平滑：R5 = 1 - S*(sin^2(...)_x + sin^2(...)_y)
R5 = 1.0 - S * (np.sin(argx)**2 + np.sin(argy)**2)
# 九点平滑：R9 = (1 - 2 S sin^2(..._x)) * (1 - 2 S sin^2(..._y))
R9 = (1.0 - 2.0 * S * np.sin(argx)**2) * (1.0 - 2.0 * S * np.sin(argy)**2)

# ———— 准备绘图数据：波长从 2 到 36 ————
x      = L[1:]          # 2,3,…,36
R5_p   = R5[1:]         # 对应的响应值
R9_p   = R9[1:]
diff_p = R5_p - R9_p    # R5 - R9

# ———— 开始绘图 ————
plt.figure(figsize=(8, 5))

# 五点平滑曲线
plt.plot(x, R5_p,   'k-',   linewidth=1.5, label=r'$R_{5}(1/2,L)$')
# 九点平滑曲线
plt.plot(x, R9_p,   'k--',  linewidth=1.5, label=r'$R_{9}(-1/2,L)$')
# 差值曲线
plt.plot(x, diff_p, 'k-.',  linewidth=1.5, label=r'$R_{5}(1/2,L)\!-\!R_{9}(-1/2,L)$')

# 坐标范围与刻度
plt.xlim(0, 36)
plt.ylim(-0.5, 1)
plt.xticks(
    [0,2,4,8,12,16,20,24,28,32,36],
    ['0','2','4','8','12','16','20','24','28','32','36']
)
plt.yticks([-0.5, 0, 0.5, 1], ['-0.5','0','0.5','1'])

# 标签与标题
plt.title('五点平滑、九点平滑的响应函数随波长的变化', fontsize=14)
plt.xlabel(r'$L/\Delta x$', fontsize=12)
plt.ylabel('响应函数 R', fontsize=12)

# 栅格线（点状）
plt.grid(linestyle=':', linewidth=0.5)

# 图例
plt.legend(loc='upper right', fontsize=12)

# 显示
plt.tight_layout()
plt.show()
