import torch

def matrix_to_bitmap(matrix):
    """
    将8x8矩阵转换为uint64的bitmap
    :param matrix: 8x8的二维数组或张量
    :return: uint64数值
    """
    assert matrix.shape == (8, 8), "必须是8x8矩阵"
    bitmap = 0
    for i in range(8):
        for j in range(8):
            if matrix[i][j] != 0:
                # 行优先计算比特位置：i*8 + j
                bitmap |= 1 << (i * 8 + j)
    return bitmap

def bitmap_to_matrix(bitmap):
    """
    将uint64 bitmap转换为8x8矩阵
    :param bitmap: uint64数值
    :return: 8x8矩阵张量
    """
    matrix = torch.zeros((8, 8), dtype=torch.uint8)
    for pos in range(64):
        if bitmap & (1 << pos):
            i, j = pos // 8, pos % 8  # 计算行列位置
            matrix[i][j] = 1
    return matrix

def print_formats(bitmap):
    """打印数值的各种格式"""
    print(f" 十进制:\t{bitmap}")
    
    bin_str = bin(bitmap)[2:]  # 去掉'0b'前缀
    print(f" 二进制:\t", ' '.join([bin_str[i:i+8] for i in range(0, 64, 8)])) 
    # print(f"二进制: {bin(bitmap)}")
    
    # 每4位一组显示十六进制
    hex_str = f"{bitmap:016X}"
    print(" 十六进制:\t", ' '.join([hex_str[i:i+4] for i in range(0, 16, 4)]))

# 示例1：创建下三角矩阵（含对角线）
def create_lower_triangular():
    matrix = torch.zeros((8, 8), dtype=torch.uint8)
    for i in range(8):
        for j in range(8):
            if i >= j:  # 下三角部分（含对角线）
                matrix[i][j] = 1
    return matrix

# 示例2：从数值重建矩阵
def demo_reverse(bitmap):
    print("\n=== 数值转矩阵演示 ===")
    print_formats(bitmap)
    print("\n对应的矩阵")
    matrix = bitmap_to_matrix(bitmap)
    for row in matrix:
        print(' '.join(map(str, row.tolist())))

if __name__ == "__main__":
    # 演示1：下三角矩阵转换
    print("=== 下三角矩阵转换演示 ===")
    lower_tri = create_lower_triangular()
    print("原始下三角矩阵：")
    for row in lower_tri:
        print(' '.join(map(str, row.tolist())))
    
    bitmap = matrix_to_bitmap(lower_tri)
    print("\n转换结果")
    print_formats(bitmap)

    # 演示2：给定数值重建矩阵
    demo_reverse(0xFFFF000000000000)  # 上半部分全1
    demo_reverse(0x8040201008040201)  # 对角线