import numpy as np
from tqdm import tqdm

def nine_point_smooth(field, smooth_coef, nx, ny):
    """
    九点平滑函数
    参数:
    field: 输入的要素场（二维数组）
    smooth_coef: 平滑系数
    nx: 经向格点数
    ny: 纬向格点数
    
    返回:
    smoothed_field: 平滑后的场
    """
    # 创建输出数组
    smoothed_field = np.zeros((nx, ny))
    
    # 为了处理边界，创建一个扩展的数组
    extended_field = np.pad(field, ((1, 1), (1, 1)), mode='edge')
    
    # 使用tqdm显示进度
    print("正在进行九点平滑...")
    for i in tqdm(range(1, nx+1)):
        for j in range(1, ny+1):
            # 九点权重
            center = extended_field[i, j]
            adjacent = (extended_field[i-1, j] + extended_field[i+1, j] + 
                       extended_field[i, j-1] + extended_field[i, j+1])
            corner = (extended_field[i-1, j-1] + extended_field[i-1, j+1] + 
                     extended_field[i+1, j-1] + extended_field[i+1, j+1])
            
            # 计算平滑后的值
            smoothed_value = (1 - smooth_coef) * center + \
                           smooth_coef * (0.5 * adjacent + 0.25 * corner) / 3
            
            # 存储结果
            smoothed_field[i-1, j-1] = smoothed_value
            
    return smoothed_field

# 测试代码
if __name__ == "__main__":
    # 创建示例数据
    nx, ny = 10, 10
    test_field = np.random.rand(nx, ny)
    smooth_coef = 0.5
    
    # 进行平滑
    result = nine_point_smooth(test_field, smooth_coef, nx, ny)
    
    # 输出原始数据和平滑后的数据对比
    print("\n原始数据示例:")
    print(test_field[0:3, 0:3])
    print("\n平滑后数据示例:")
    print(result[0:3, 0:3])