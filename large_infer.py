import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import osgeo.gdal as gdal
import torch
import numpy as np
from tqdm import tqdm

from networks.MHNet import MHNet

def base_predict(file_path, save_path, net, winsize=256, buffersize=128, thresh=0.5):
    """
    对大幅遥感影像进行滑窗推理并保存为单波段掩膜GeoTIFF（0/255，保留GeoTransform与Projection）
    """
    os.makedirs(save_path, exist_ok=True)

    original = gdal.Open(file_path, gdal.GA_ReadOnly)
    if original is None:
        raise FileNotFoundError(f"Cannot open: {file_path}")

    bands = original.RasterCount
    # 取前三个波段；若不足3个则尽可能多取
    read_bands = min(3, bands)
    img = np.stack([original.GetRasterBand(i+1).ReadAsArray() for i in range(read_bands)], axis=0)

    # 若只有1或2个波段，简单重复到3通道以兼容训练/归一化流程
    if img.shape[0] < 3:
        img = np.tile(img, (3 // img.shape[0] + 1, 1, 1))[:3, ...]

    row, col = img.shape[1], img.shape[2]
    geo_transform = original.GetGeoTransform()
    projection = original.GetProjection()

    # 对称填充，保证整除滑窗 + 边界缓冲
    pad = np.pad(
        img,
        ((0, 0),
         (winsize, int(winsize + (np.ceil(row / winsize) * winsize) - row) + buffersize),
         (winsize, int(winsize + (np.ceil(col / winsize) * winsize) - col) + buffersize)),
        mode='symmetric'
    )

    nrow, ncol = pad.shape[1], pad.shape[2]
    result = torch.zeros([nrow, ncol], device='cuda')

    print('Image size with padding:', pad.shape)

    num_r = int(row / winsize) + 1
    num_c = int(col / winsize) + 1

    net.eval()
    with torch.no_grad():
        for r in tqdm(range(num_r), desc=f'Rows', leave=True):
            r0 = int((r + 1) * winsize) - buffersize
            r1 = int((r + 2) * winsize) + buffersize
            for c in range(num_c):
                c0 = int((c + 1) * winsize) - buffersize
                c1 = int((c + 2) * winsize) + buffersize

                tile = pad[:, r0:r1, c0:c1]

                if tile.max() == 0:
                    # 全黑块直接跳过
                    result[int((r + 1) * winsize):int((r + 2) * winsize),
                           int((c + 1) * winsize):int((c + 2) * winsize)] = 0
                    continue

                # 归一化到[-1,1]
                input_tensor = torch.tensor(tile, dtype=torch.float32, device='cuda').unsqueeze(0) / 255.0
                input_tensor = (input_tensor - 0.5) / 0.5

                out_dict = net(input_tensor)
                output = out_dict['label_pred'].squeeze()

                # 二值化到0/255
                output = (output > thresh).float() * 255.0

                result[int((r + 1) * winsize):int((r + 2) * winsize),
                       int((c + 1) * winsize):int((c + 2) * winsize)] = output[
                            buffersize:buffersize + winsize,
                            buffersize:buffersize + winsize
                       ]

    # 去掉padding
    result = result[winsize:winsize + row, winsize:winsize + col]
    result_np = result.detach().cpu().numpy().astype(np.uint8)

    # 保存为单波段Byte型GeoTIFF
    imgname = os.path.splitext(os.path.basename(file_path))[0] + '.tif'
    output_file = os.path.join(save_path, imgname)
    print(f'Saving to: {output_file}')

    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        output_file,
        col,  # width
        row,  # height
        1,    # single band
        gdal.GDT_Byte
    )
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(result_np)
    dataset.FlushCache()
    dataset = None

    print('Done.')
    return output_file


if __name__ == '__main__':
    # -------- 基本配置 --------
    winsize = 256
    buffersize = 128
    thresh = 0.5

    # 输入与输出路径（按你的要求）
    image_path = "/data/Mozambique_Benga Coal Mine.tif"
    save_path  = "/code/experiments"

    # 权重与模型
    weight_file = "/code/experiments/MHNet/weights/MHNet_50.th"

    # 可选：加速
    torch.backends.cudnn.benchmark = True

    # 构建并加载模型
    net = MHNet(num_classes=1)
    net = net.cuda()
    # 多卡兼容
    # if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))

    state = torch.load(weight_file, map_location='cuda')
    net.load_state_dict(state)
    print('Model loaded successfully.')

    # 仅推理一张大图
    _ = base_predict(
        file_path=image_path,
        save_path=save_path,
        net=net,
        winsize=winsize,
        buffersize=buffersize,
        thresh=thresh
    )
