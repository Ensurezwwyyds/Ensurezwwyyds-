import numpy as np

def magnification_factor_and_coriolis_parameter(proj, In, Jn, d):
    """
    计算放大系数和科里奥利参数
    
    参数:
    proj -- 投影类型，例如 'stereographic'
    In, Jn -- 网格坐标
    d -- 网格间距
    
    返回:
    m -- 放大系数
    f -- 科里奥利参数
    """
    a = 6371  # 地球半径
    omega = 7.292e-5  # 地球自转角速度
    
    if proj == 'stereographic':
        le = 11888.45
        l = np.sqrt((In**2 + Jn**2) * d**2)
        m = (2 + np.sqrt(3)) / 2 / (1 + ((le**2 - l**2) / (le**2 + l**2)))
        f = 2 * omega * ((le**2 - l**2) / (le**2 + l**2))
    
    return m, f

# 示例使用
if __name__ == "__main__":
    # 示例参数
    proj = 'stereographic'
    In = 10
    Jn = 10
    d = 100
    
    # 计算放大系数和科里奥利参数
    m, f = magnification_factor_and_coriolis_parameter(proj, In, Jn, d)
    
    print(f"放大系数 (m): {m}")
    print(f"科里奥利参数 (f): {f}")