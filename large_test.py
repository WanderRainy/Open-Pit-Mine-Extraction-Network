import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import osgeo.gdal as gdal
import torch
import numpy as np
import cv2
import os
import time
from tqdm import tqdm  # 进度条
from torch.utils.data import DataLoader
from networks.dinknet import LinkNet50, BAM_LinkNet50, DinkNet50
from networks.MHNet import MHNet

import rasterio
from rasterio.transform import from_origin
import tifffile as tiff
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score, BinaryPrecision, BinaryRecall
    
def base_predict(file_path, save_path, net, winsize=512, buffersize=192):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    original = gdal.Open(file_path, gdal.GA_ReadOnly)
    img = original.ReadAsArray()[:3]  # 获取前三个波段
    row, col = img.shape[1], img.shape[2]
    geo_transform = original.GetGeoTransform()
    projection = original.GetProjection()

    pad = np.pad(img, ((0, 0), 
                       (winsize, int(winsize + ((np.ceil(row / winsize)) * winsize) - row) + buffersize),
                       (winsize, int(winsize + ((np.ceil(col / winsize)) * winsize) - col) + buffersize)), 
                 mode='symmetric')

    nrow, ncol = pad.shape[1], pad.shape[2]
    result = torch.zeros([nrow, ncol]).cuda()

    print('Image size:', pad.shape)
    
    num_steps = int(row / winsize) + 1
    with torch.no_grad():
        for r in tqdm(range(num_steps), desc=f'Processing rows for {os.path.basename(file_path)}', leave=True):
            for c in range(int(col / winsize) + 1):
                tem = pad[:, int((r + 1) * winsize) - buffersize:int((r + 2) * winsize) + buffersize,
                          int((c + 1) * winsize) - buffersize:int((c + 2) * winsize) + buffersize]

                if tem.max() == 0:
                    result[int((r + 1) * winsize):int((r + 2) * winsize),
                           int((c + 1) * winsize):int((c + 2) * winsize)] = 0
                    continue

                input_tensor = torch.Tensor(np.array([tem], np.float32) / 255.0).cuda()
                input_tensor = (input_tensor - 0.5) / 0.5
                # output_dict = net(input_tensor)
                output = net(input_tensor)['label_pred'].squeeze()
                output = (output > 0.5).float() * 255

                result[int((r + 1) * winsize):int((r + 2) * winsize),
                       int((c + 1) * winsize):int((c + 2) * winsize)] = output[
                                                                        buffersize:buffersize + winsize,
                                                                        buffersize:buffersize + winsize]

    result = result[winsize:winsize + row, winsize:winsize + col]
    result = result.cpu().numpy().astype(np.uint8)

    # 保存结果为 TIFF 文件
    print(f'Creating image for {os.path.basename(file_path)}...')
    imgname = os.path.basename(file_path).split('.')[0] + '.tif'
    output_file = os.path.join(save_path, imgname)

    height, width = result.shape

    # 使用 GDAL 创建一个新的 TIFF 文件
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        output_file,                # 输出文件路径
        width,                       # 图像的宽度
        height,                      # 图像的高度
        1,                           # 单波段
        gdal.GDT_Float32,           # 数据类型（根据需要选择，例：Float32，Int16，等）
    )

    # 设置地理转换信息（GeoTransform）
    dataset.SetGeoTransform(geo_transform)

    # 设置投影信息（Projection）
    dataset.SetProjection(projection)

    # 将 NumPy 数组写入 TIFF 文件
    dataset.GetRasterBand(1).WriteArray(result)

    # 清理并关闭文件
    dataset = None  # 关闭数据集并保存文件

    return result / 255


if __name__ == '__main__':
    winsize = 256
    buffersize = 128
    # root = '/data1/wyx24/mining/merge/'
    save_path = "/MHNet/large_results/"
    os.makedirs(save_path, exist_ok=True)
    weight_file = "/MHNet/weights/MHNet_50.th"
    modelname = os.path.basename(weight_file).split('.')[0]
    
    jaccard = BinaryJaccardIndex().cuda()
    f1_score = BinaryF1Score().cuda()
    precision = BinaryPrecision().cuda()
    recall = BinaryRecall().cuda()
    # 加载模型
    net = MHNet(num_classes=1)
    net = net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.load_state_dict(torch.load(weight_file))
    net.eval()
    print('Model loaded successfully.')

    # 定义影像和标签路径
    image_paths = [
        "/large/test/im19/19年50cm_WGS84jwd__27.tif",
        "/large/test/im19/19年50cm_WGS84jwd__28.tif",
        "/large/test/im19/19年50cm_WGS84jwd__29.tif",
        "/large/test/im19/19年50cm_WGS84jwd__30.tif",
        "/large/test/im19/19年50cm_WGS84jwd__31.tif",
        "/large/test/im19/19年50cm_WGS84jwd__32.tif",
        "/large/test/im19/19年50cm_WGS84jwd__33.tif",
    ]

    label_paths = [
        "/large/test/mask19/19年50cm_WGS84jwd__27.tif",
        "/large/test/mask19/19年50cm_WGS84jwd__28.tif",
        "/large/test/mask19/19年50cm_WGS84jwd__29.tif",
        "/large/test/mask19/19年50cm_WGS84jwd__30.tif",
        "/large/test/mask19/19年50cm_WGS84jwd__31.tif",
        "/large/test/mask19/19年50cm_WGS84jwd__32.tif",
        "/large/test/mask19/19年50cm_WGS84jwd__33.tif",
    ]


    # 遍历每个影像并进行预测和评估
    for img_path, label_path in zip(image_paths, label_paths):
        img_name = os.path.basename(img_path).split('.')[0]

        # 预测结果
        pred = base_predict(img_path, save_path, net)
        pred = np.array(pred, dtype=np.uint8)

        # 读取标签
        # label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label_array = tiff.imread(label_path)
        if label_array.max() == 255:
            label = label_array / 255
        label = np.array(label_array, dtype=np.uint8)

        # 计算评估指标
        pred = torch.tensor(pred).cuda()
        label = torch.tensor(label).cuda()
        jaccard(pred, label)
        f1_score(pred, label)
        precision(pred, label)
        recall(pred, label)
        print(f'Processing finished for {img_name}!')
        iou = jaccard.compute()
        f1 = f1_score.compute()
        prec = precision.compute()
        rec = recall.compute()
            # 保存实验结果
        with open(os.path.join(save_path, img_name + '_metric.log'), 'a') as metric_file:
            metric_file.write(f'--------{time.strftime("%Y-%m-%d %H:%M", time.localtime())}--------\n')
            metric_file.write(f'{weight_file}\n')
            metric_file.write(f'pa: {prec}\n')
            metric_file.write(f'pr: {rec}\n')
            metric_file.write(f'f1: {f1}\n')
            metric_file.write(f'iou: {iou}\n')
            metric_file.write(f'win_size: {winsize}, buffersize: {buffersize}\n')

        
