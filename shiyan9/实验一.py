# -*- coding: utf-8 -*-
import numpy as np

def time_smooth(u_prev, u_curr, u_next,
                v_prev, v_curr, v_next,
                z_prev, z_curr, z_next,
                S):
    """
    三点时间平滑子程序

    参数
    ----
    u_prev, u_curr, u_next : ndarray
        前、中、后三个时次的 u 分量场（形状相同）
    v_prev, v_curr, v_next : ndarray
        前、中、后三个时次的 v 分量场（形状相同）
    z_prev, z_curr, z_next : ndarray
        前、中、后三个时次的位势高度场（形状相同）
    S : float
        平滑系数，取值范围 [0, 0.5]。例如 S=0.25 时，对应的权重为
        [S, 1-2S, S] = [0.25, 0.5, 0.25]

    返回
    ----
    u_s, v_s, z_s : ndarray
        平滑后的三个场，形状与输入相同
    """
    if not (0 <= S <= 0.5):
        raise ValueError("平滑系数 S 应在 [0,0.5] 范围内")
    w0 = 1.0 - 2.0 * S  # 中心权重
    # 对一个场做三点平滑
    def _smooth(f_prev, f_cur, f_next):
        return S * f_prev + w0 * f_cur + S * f_next

    u_s = _smooth(u_prev, u_curr, u_next)
    v_s = _smooth(v_prev, v_curr, v_next)
    z_s = _smooth(z_prev, z_curr, z_next)

    return u_s, v_s, z_s

# ———— 示例用法 ————
if __name__ == "__main__":
    # 假设我们有三个时次的 u, v, z 数据，以下用随机数据举例
    shape = (50, 100)      # 比如纬度50×经度100网格
    u_prev = np.random.randn(*shape)
    u_curr = np.random.randn(*shape)
    u_next = np.random.randn(*shape)
    v_prev = np.random.randn(*shape)
    v_curr = np.random.randn(*shape)
    v_next = np.random.randn(*shape)
    z_prev = np.random.randn(*shape)
    z_curr = np.random.randn(*shape)
    z_next = np.random.randn(*shape)

    S = 0.25  # 平滑系数
    u_s, v_s, z_s = time_smooth(
        u_prev, u_curr, u_next,
        v_prev, v_curr, v_next,
        z_prev, z_curr, z_next,
        S
    )

    # 打印一下均值以示成功
    print("u_s mean:", np.nanmean(u_s))
    print("v_s mean:", np.nanmean(v_s))
    print("z_s mean:", np.nanmean(z_s))
