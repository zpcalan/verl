import sys
import os
import numpy as np

def compare_tensors(tensor1, tensor2, tolerance=1e-6):
    """
    对比两个tensor
    
    参数:
    tensor1 (np.ndarray): 第一个tensor
    tensor2 (np.ndarray): 第二个tensor
    tolerance (float): 数值比较的容差
    
    返回:
    dict: 包含比较结果的字典
    """
    results = {
        'shape_equal': False,
        'dtype_equal': False,
        'exact_equal': False,
        'close_equal': False,
        'max_abs_diff': None,
        'mean_abs_diff': None,
        'relative_diff': None,
        'mse': None,
        'cosine_similarity': None,
        'tolerance': tolerance
    }
    
    # 1. 检查shape是否相同
    shape1 = tensor1.shape
    shape2 = tensor2.shape
    results['shape_equal'] = shape1 == shape2
    results['shape1'] = shape1
    results['shape2'] = shape2
    results['max_1'] = np.max(tensor1)
    results['max_2'] = np.max(tensor2)
    results['min_1'] = np.min(tensor1)
    results['min_2'] = np.min(tensor2)
    
    # 如果shape不同，直接返回结果
    if not results['shape_equal']:
        return results
    
    # 2. 检查dtype是否相同
    dtype1 = tensor1.dtype
    dtype2 = tensor2.dtype
    results['dtype_equal'] = dtype1 == dtype2
    results['dtype1'] = dtype1
    results['dtype2'] = dtype2
    
    # 3. 检查是否完全相等（适用于整数类型）
    results['exact_equal'] = np.array_equal(tensor1, tensor2)
    
    # 4. 检查是否在容差范围内相等（适用于浮点数）
    if np.issubdtype(tensor1.dtype, np.floating) or np.issubdtype(tensor2.dtype, np.floating):
        results['close_equal'] = np.allclose(tensor1, tensor2, rtol=tolerance, atol=tolerance)
    
    # 5. 计算各种差异指标
    # 绝对差异
    abs_diff = np.abs(tensor1 - tensor2)
    results['max_abs_diff'] = np.max(abs_diff)
    results['mean_abs_diff'] = np.mean(abs_diff)
    
    # 相对差异（避免除以0）
    if np.any(tensor1 != 0):
        relative_diff = np.abs((tensor1 - tensor2) / (np.abs(tensor1) + 1e-10))
        results['relative_diff'] = np.mean(relative_diff)
    
    # 均方误差
    results['mse'] = np.mean((tensor1 - tensor2) ** 2)
    
    # 余弦相似度（对于向量化的tensor）
    try:
        # 将tensor展平为向量
        flat1 = tensor1.flatten()
        flat2 = tensor2.flatten()
        
        # 计算点积和模长
        dot_product = np.dot(flat1, flat2)
        norm1 = np.linalg.norm(flat1)
        norm2 = np.linalg.norm(flat2)
        
        if norm1 > 0 and norm2 > 0:
            results['cosine_similarity'] = dot_product / (norm1 * norm2)
        else:
            results['cosine_similarity'] = 0.0
    except:
        results['cosine_similarity'] = None
    
    return results

def print_comparison_results(results, verbose=False):
    """
    打印对比结果
    
    参数:
    results (dict): 对比结果
    verbose (bool): 是否显示详细信息
    """
    print("=" * 60)
    print("TENSOR 对比结果")
    print("=" * 60)
    
    # 基本信息
    print(f"Tensor 1 Shape: {results['shape1']}")
    print(f"Tensor 2 Shape: {results['shape2']}")
    print(f"Shape 相等: {'✓' if results['shape_equal'] else '✗'}")
    
    if results['shape_equal']:
        print(f"Tensor 1 Max: {results['max_1']}")
        print(f"Tensor 2 Max: {results['max_2']}")
        print(f"Tensor 1 Min: {results['min_1']}")
        print(f"Tensor 2 Min: {results['min_2']}")
        print(f"Dtype 相等: {'✓' if results['dtype_equal'] else '✗'}")
        print(f"完全相等 (exact): {'✓' if results['exact_equal'] else '✗'}")
        
        if 'close_equal' in results:
            print(f"容差相等 (tolerance={results['tolerance']}): {'✓' if results['close_equal'] else '✗'}")
        
        print("\n差异统计:")
        print(f"最大绝对差异: {results['max_abs_diff']:.6e}")
        print(f"平均绝对差异: {results['mean_abs_diff']:.6e}")
        
        if results['relative_diff'] is not None:
            print(f"平均相对差异: {results['relative_diff']:.6e}")
        
        print(f"均方误差 (MSE): {results['mse']:.6e}")
        
        if results['cosine_similarity'] is not None:
            print(f"余弦相似度: {results['cosine_similarity']:.6f}")
            
            # 解释余弦相似度
            cos_sim = results['cosine_similarity']
            if cos_sim > 0.99:
                print("  → 向量方向几乎相同")
            elif cos_sim > 0.9:
                print("  → 向量方向高度相似")
            elif cos_sim > 0.7:
                print("  → 向量方向相似")
            elif cos_sim > 0.5:
                print("  → 向量方向有一定相似性")
            elif cos_sim > 0:
                print("  → 向量方向有较小相似性")
            elif cos_sim == 0:
                print("  → 向量正交")
            else:
                print("  → 向量方向相反")
    
    if verbose:
        print("\n详细形状信息:")
        print(f"Tensor 1 维度数: {len(results['shape1'])}")
        print(f"Tensor 2 维度数: {len(results['shape2'])}")
        
        if results['shape_equal']:
            print(f"Tensor 元素总数: {np.prod(results['shape1'])}")
            
            # 内存使用估计
            elem_size1 = np.dtype(results['dtype1']).itemsize
            elem_size2 = np.dtype(results['dtype2']).itemsize
            memory1 = np.prod(results['shape1']) * elem_size1
            memory2 = np.prod(results['shape2']) * elem_size2
            print(f"Tensor 1 内存占用: {memory1 / 1024:.2f} KB")
            print(f"Tensor 2 内存占用: {memory2 / 1024:.2f} KB")
    
    print("=" * 60)

def process_files(file1_path, file2_path):
    """
    处理两个文件的函数
    您可以在此实现具体的业务逻辑
    
    参数:
    file1_path (str): 第一个文件的路径
    file2_path (str): 第二个文件的路径
    """
    input_1 = np.load(file1_path)
    input_2 = np.load(file2_path)
    print(f"{file1_path}: {input_1}")
    print(f"{file2_path}: {input_2}")
    results = compare_tensors(input_1, input_2)
    print_comparison_results(results)

def main():
    # 检查命令行参数数量
    if len(sys.argv) != 3:
        print("用法: python file_processor.py <文件1路径> <文件2路径>")
        print("示例: python file_processor.py ./data/file1.txt ./data/file2.txt")
        sys.exit(1)
    
    # 获取文件路径参数
    file1_path = sys.argv[1]
    file2_path = sys.argv[2]
    
    # 处理文件
    if process_files(file1_path, file2_path):
        print("文件处理完成")
    else:
        print("文件处理失败")

if __name__ == "__main__":
    main()