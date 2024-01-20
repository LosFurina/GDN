import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import os
import torch.nn as nn
import time
from util.time import *
from util.env import *

import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F

from util.data import *
from util.preprocess import *


def test(model, dataloader):
    # test
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()

    test_loss_list = []
    now = time.time()

    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    test_len = len(dataloader)

    model.eval()

    i = 0
    acu_loss = 0
    for x, y, labels, edge_index in dataloader:
        x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]

        with torch.no_grad():
            predicted = model(x, edge_index).float().to(device)

            loss = loss_func(predicted, y)

            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)

        test_loss_list.append(loss.item())
        acu_loss += loss.item()

        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))

    test_predicted_list = t_test_predicted_list.tolist()
    test_ground_list = t_test_ground_list.tolist()
    test_labels_list = t_test_labels_list.tolist()

    avg_loss = sum(test_loss_list) / len(test_loss_list)

    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type=int, default=128)
    parser.add_argument('-epoch', help='train epoch', type=int, default=100)
    parser.add_argument('-slide_win', help='slide_win', type=int, default=15)
    parser.add_argument('-dim', help='dimension', type=int, default=64)
    parser.add_argument('-slide_stride', help='slide_stride', type=int, default=5)
    parser.add_argument('-save_path_pattern', help='save path pattern', type=str, default='')
    parser.add_argument('-dataset', help='wadi / swat /msl / aidd', type=str, default='aidd')
    parser.add_argument('-device', help='cuda / cpu', type=str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type=int, default=0)
    parser.add_argument('-comment', help='experiment comment', type=str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type=int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type=int, default=256)
    parser.add_argument('-decay', help='decay', type=float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type=float, default=0.1)
    parser.add_argument('-topk', help='topk num', type=int, default=20)
    parser.add_argument('-report', help='best / val', type=str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type=str, default='')

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
    }

    env_config = {
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path
    }
    from main import Main
    main = Main(train_config, env_config, debug=False)

    main.model.load_state_dict(torch.load("/pretrained/best_01-10-15-14-00-MA.pt"))
    best_model = main.model.to(main.device)

    _, main.test_result = test(best_model, main.test_dataloader)
    _, main.val_result = test(best_model, main.val_dataloader)
    pred = np.array(main.test_result[0])
    ground = np.array(main.test_result[1])

    # 获取数组的列数
    num_columns = pred.shape[1]

    # 循环绘制每一列的线图
    os.makedirs("./data/ma/pic", exist_ok=True)
    for ft_ith in range(num_columns):
        # Create a figure and axis for the current iteration
        fig, (ax_upper, ax_lower) = plt.subplots(2, 1, figsize=(8, 6))

        # Plot the prediction and ground truth in the upper half
        ax_upper.plot(pred[:, ft_ith], label=f'Prediction_{ft_ith}', color='blue')
        ax_upper.plot(ground[:, ft_ith], label=f'Ground Truth_{ft_ith}', color='orange')
        ax_upper.set_xlabel('Index')
        ax_upper.set_ylabel('Values')
        ax_upper.legend()

        # Calculate residuals
        residuals = ground[:, ft_ith] - pred[:, ft_ith]

        # Plot residuals in the lower half
        ax_lower.plot(residuals, label=f'Residuals_{ft_ith}', color='green')
        ax_lower.set_xlabel('Index')
        ax_lower.set_ylabel('Residuals')
        ax_lower.legend()

        # Save the combined plot as a PNG file
        path_combined = f"./data/ma/pic/combined_plot_{ft_ith}.png"
        plt.savefig(path_combined)

        # Close the figure for the current iteration
        plt.close(fig)
