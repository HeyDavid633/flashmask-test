import torch
import numpy as np

def matrix_to_bitmaps(matrix, block_size=(8,8)):
    """
    将大矩阵分块转换为uint64数组
    :param matrix: 大矩阵（尺寸需为block_size的整数倍）
    :param block_size: 分块尺寸 (m,n)
    :return: uint64数组（按列优先顺序排列）
    """
    m, n = block_size
    assert matrix.shape[0] % m == 0 and matrix.shape[1] % n == 0, "矩阵尺寸必须是分块尺寸的整数倍"
    
    blocks = []
    # 列优先遍历分块
    for j in range(0, matrix.shape[1], n):
        for i in range(0, matrix.shape[0], m):
            block = matrix[i:i+m, j:j+n]
            bitmap = 0
            # 行优先编码子矩阵
            for bi in range(m):
                for bj in range(n):
                    if block[bi][bj] != 0:
                        bitmap |= 1 << (bi * n + bj)
            blocks.append(bitmap)
    return np.array(blocks, dtype=np.uint64)

def bitmaps_to_matrix(bitmaps, matrix_shape, block_size=(8,8)):
    """
    将uint64数组恢复为原始矩阵
    :param bitmaps: uint64数组
    :param matrix_shape: 目标矩阵尺寸
    :param block_size: 分块尺寸 (m,n)
    :return: 重建的矩阵
    """
    m, n = block_size
    matrix = torch.zeros(matrix_shape, dtype=torch.uint8)
    bitmap_idx = 0
    # 列优先重建分块
    for j in range(0, matrix.shape[1], n):
        for i in range(0, matrix.shape[0], m):
            bitmap = bitmaps[bitmap_idx]
            # 行优先解码子矩阵
            for pos in range(m*n):
                if bitmap & (np.uint64(1) << np.uint64(pos)):
                    bi, bj = pos // n, pos % n
                    matrix[i+bi][j+bj] = 1
            bitmap_idx += 1
    return matrix

def print_bitmaps(bitmaps):
    """格式化输出bitmap数组"""
    print("[\n" + ",\n".join([f"  {hex(b)} (dec:{b})" for b in bitmaps]) + "\n]")
    
    
# 创建16x16下三角矩阵
def create_16x16_lower_triangular():
    matrix = torch.zeros((16, 16), dtype=torch.uint8)
    for i in range(16):
        for j in range(16):
            if i >= j:
                matrix[i][j] = 1
    return matrix

if __name__ == "__main__":
    # 演示1：16x16下三角矩阵编码
    big_matrix = create_16x16_lower_triangular()
    print("原始16x16下三角矩阵：")
    for i in range(0, 16, 8):
        for row in big_matrix[i:i+8]:
            print(' '.join(map(str, row.tolist())))
        print("-"*32)
    
    bitmaps = matrix_to_bitmaps(big_matrix)
    print("\n编码结果（4个uint64）：")
    print_bitmaps(bitmaps)

    # 演示2：验证解码
    reconstructed = bitmaps_to_matrix(bitmaps, (16,16))
    print("\n解码后矩阵验证：", torch.allclose(big_matrix, reconstructed))