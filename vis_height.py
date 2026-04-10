import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from concurrent.futures import ProcessPoolExecutor, as_completed

# 路径设置
image_dir = "/data1/Miningset/image_v2/test/"  # 高分辨率遥感影像路径
height_dir = "/data1/yry22/temp/Mine_rsl/code/experiments/MHNet/height_results"  # 高度数据路径
output_dir = "/data1/yry22/temp/Mine_rsl/code/experiments/vis_height"  # 可视化结果保存路径

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 可视化函数
def visualize_image(image_name):
    if image_name.endswith(".tif"):  # 仅处理 .tif 文件
        try:
            # 加载遥感影像
            image_path = os.path.join(image_dir, image_name)
            with rasterio.open(image_path) as src:
                image = src.read([1, 2, 3])  # 读取 RGB 波段
                image = np.transpose(image, (1, 2, 0))  # 转换为 (H, W, C) 格式

            # 加载高度数据
            height_path = os.path.join(height_dir, image_name)
            with rasterio.open(height_path) as src:
                height = src.read(1)  # 读取第一个波段（假设高度数据是单波段）

            # 可视化
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 一行两列

            # 显示遥感影像
            ax1.imshow(image)
            ax1.set_title("Remote Sensing Image")
            ax1.axis("off")

            # 显示高度数据，使用内置颜色条
            height_plot = ax2.imshow(height, cmap=plt.cm.viridis, vmin=0, vmax=1)  # 使用 viridis 颜色条
            ax2.set_title("Height Map")
            ax2.axis("off")

            # 去掉颜色条
            # plt.colorbar(height_plot, ax=ax2, fraction=0.046, pad=0.04)  # 注释掉这行

            # 保存结果
            output_path = os.path.join(output_dir, image_name.replace(".tif", ".png"))  # 保存为 PNG 格式
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close()

            print(f"Saved visualization for {image_name} to {output_path}")
            return True
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            return False
    return False

# 并行处理
def parallel_processing(image_names, max_workers=4):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(visualize_image, name) for name in image_names]
        for future in as_completed(futures):
            future.result()  # 等待任务完成

# 主函数
if __name__ == "__main__":
    # 获取所有影像文件名
    image_names = [name for name in os.listdir(image_dir) if name.endswith(".tif")]

    # 并行处理
    parallel_processing(image_names, max_workers=8)  # 设置最大并行数为 8