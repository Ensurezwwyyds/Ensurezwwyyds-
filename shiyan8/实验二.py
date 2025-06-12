import numpy as np
from tqdm import tqdm

def five_point_smooth(field, smooth_coef, nx, ny, smooth_type):
    """
    五点平滑函数
    参数:
    field: 输入的要素场（二维数组）
    smooth_coef: 平滑系数
    nx: 经向格点数
    ny: 纬向格点数
    smooth_type: 平滑类型（0为正平滑，1为逆平滑）
    
    返回:
    smoothed_field: 平滑后的场
    """
    # 创建工作数组
    work = np.zeros_like(field)
    
    # 复制输入数组以保持原始数据不变
    result = field.copy()
    
    # 正平滑
    if smooth_type == 0:
        print("正在进行正平滑...")
        for i in tqdm(range(1, nx-1)):
            for j in range(1, ny-1):
                # 计算五点平滑
                center = result[i, j]
                adjacent_sum = (result[i-1, j] + result[i+1, j] + 
                              result[i, j-1] + result[i, j+1])
                work[i, j] = center + smooth_coef * (adjacent_sum - 4.0 * center) / 4.0
        
        # 更新结果
        result[1:nx-1, 1:ny-1] = work[1:nx-1, 1:ny-1]
    
    # 逆平滑
    elif smooth_type == 1:
        print("正在进行逆平滑...")
        for i in tqdm(range(1, nx-1)):
            for j in range(1, ny-1):
                # 计算五点逆平滑
                center = result[i, j]
                adjacent_sum = (result[i-1, j] + result[i+1, j] + 
                              result[i, j-1] + result[i, j+1])
                work[i, j] = center - smooth_coef * (adjacent_sum - 4.0 * center) / 4.0
        
        # 更新结果
        result[1:nx-1, 1:ny-1] = work[1:nx-1, 1:ny-1]
    
    return result

# 测试代码
if __name__ == "__main__":
    # 创建示例数据
    nx, ny = 10, 10
    test_field = np.random.rand(nx, ny)
    smooth_coef = 0.5
    
    # 进行正平滑
    print("\n===== 测试正平滑 =====")
    result_forward = five_point_smooth(test_field, smooth_coef, nx, ny, 0)
    
    # 进行逆平滑
    print("\n===== 测试逆平滑 =====")
    result_backward = five_point_smooth(test_field, smooth_coef, nx, ny, 1)
    
    # 输出结果示例
    print("\n原始数据示例:")
    print(test_field[1:4, 1:4])
    print("\n正平滑后数据示例:")
    print(result_forward[1:4, 1:4])
    print("\n逆平滑后数据示例:")
    print(result_backward[1:4, 1:4])