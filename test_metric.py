import os
import numpy as np
import cv2
import time
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
import copy
import tifffile as tiff

def load_dataset(dataset_name,crop_size=512):
    if dataset_name == 'Mine':
        dataset = Mine_Dataset(crop_size, seed=args.seed, type='test')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return dataset

def load_model(model_name, num_classes):
    model_dict = {
    'MHNet':MHNet
}

    if model_name in model_dict:
        return model_dict[model_name](num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segmentation Test Script')
    parser.add_argument('--gpu', type=str, required=True, help='CUDA visible devices')
    parser.add_argument('--dataset', type=str, required=True, choices=['Mine'], help='Dataset to use')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['MHNet'], 
                        help='Model to use')
    parser.add_argument('--seed', type=int, default=197, help='Random seed')
    parser.add_argument('--out_pred', action='store_true', help='Output predictions')
    parser.add_argument('--exper_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--crop_size', type=int, required=True, help='Crop size (width height)')
    parser.add_argument('--num_classes', type=int, default=1, help='classes')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    import torch.utils.data as data
    from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score, BinaryPrecision, BinaryRecall
    from torchmetrics.classification import JaccardIndex, F1Score, Precision, Recall

    from networks.dinknet import LinkNet50, BAM_LinkNet50, ConvNeXt_LinkNet, SwinT_LinkNet
    from networks.MHNet import LinkNet50_HK, BAM_LinkNet50_HK, MHNet, SwinT_MHNet
    # from networks_sam.sam import SAM
    from minedataset import Mine_Dataset
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset = load_dataset(args.dataset, crop_size=args.crop_size)
    model = load_model(args.model, args.num_classes)
    
    target = f'./experiments/{args.exper_name}/'
    if not os.path.isdir(target):
        os.makedirs(target)
    os.makedirs(f'./experiments/{args.exper_name}/results/', exist_ok=True)
    model = model.cuda()
    
    # Wrap the model with DataParallel to use multiple GPUs
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(f'./experiments/{args.exper_name}/weights/{args.exper_name}.th'))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # Single batch size
        shuffle=False,
        num_workers=4)
    
    # Initialize torchmetrics metrics
    if args.num_classes==1:
        jaccard = BinaryJaccardIndex().cuda()
        f1_score = BinaryF1Score().cuda()
        precision = BinaryPrecision().cuda()
        recall = BinaryRecall().cuda()
    else:
        jaccard = JaccardIndex(task="multiclass", num_classes=args.num_classes,average=None).cuda()
        f1_score = F1Score(task="multiclass", num_classes=args.num_classes,average=None).cuda()
        precision = Precision(task="multiclass", num_classes=args.num_classes,average=None).cuda()
        recall = Recall(task="multiclass", num_classes=args.num_classes,average=None).cuda()
    
    model.eval()
    with torch.no_grad():
        for data_loader_iter in tqdm(data_loader):
            img, mask, data_name = data_loader_iter
            img = img.cuda()
            mask = mask.cuda()
            # import pdb
            # pdb.set_trace()
            pred = model(img)
            if isinstance(pred, dict):
                ## vis height results
                # pred_height = pred['height_pred'][0].cpu().numpy()
                # pred_height = np.squeeze(pred_height, axis=0)
                # pred_filename = f'{target}/height_results/{data_name[0]}'
                # cv2.imwrite(pred_filename, pred_height)

                pred = pred['label_pred']

            if args.num_classes==1:
                pred = torch.round(pred)  # Thresholding the predictions to 0 or 1
            else:
                pred = torch.argmax(pred, dim=1,keepdim=True)
            # Update metrics
            jaccard(pred, mask)
            f1_score(pred, mask)
            precision(pred, mask)
            recall(pred, mask)
            if args.out_pred:
                if args.num_classes==1:
                    predout = pred[0].cpu().numpy().astype(np.uint8) * 255
                    predout = np.squeeze(predout, axis=0)  # Squeeze to remove the channel dimension if necessary
                else:
                    predout = pred[0][0].cpu().numpy().astype(np.uint8)
                # import pdb
                # pdb.set_trace()
                pred_filename = f'{target}/results/{data_name[0]}'
                cv2.imwrite(pred_filename, predout)
                # pred = torch.unsqueeze(pred, dim=0)
            
            

    # Compute final metric values
    iou = jaccard.compute()
    f1 = f1_score.compute()
    prec = precision.compute()
    rec = recall.compute()

    # Log the metrics
    metric_file = open(f'./experiments/{args.exper_name}/{args.exper_name}_metric.log', 'a')
    metric_file.write(f'--------{time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time()))}--------\n')

    # Save the metrics for each class
    if args.num_classes == 1:  # Binary classification
        metric_file.write(f'IOU: {iou.item()}\n')
        metric_file.write(f'F1 Score: {f1.item()}\n')
        metric_file.write(f'Precision: {prec.item()}\n')
        metric_file.write(f'Recall: {rec.item()}\n')
    else:  # Multiclass classification
        for i in range(args.num_classes):
            metric_file.write(f'Class {i} - IOU: {iou[i].item()}\n')
            metric_file.write(f'Class {i} - F1 Score: {f1[i].item()}\n')
            metric_file.write(f'Class {i} - Precision: {prec[i].item()}\n')
            metric_file.write(f'Class {i} - Recall: {rec[i].item()}\n')

    print('Finish!')
    metric_file.close()
