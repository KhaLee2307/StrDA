import os
import sys
import random
import argparse

import numpy as np
from tqdm import tqdm
from PIL import ImageFile

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset

from utils.averager import Averager
from utils.criterion import FocalLoss
from utils.load_config import load_config

from source.model import BaselineClassifier
from source.stratify import DomainStratifying
from source.dataset import Pseudolabel_Dataset, hierarchical_dataset, get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy("file_system")
    

def main(args):
    dashed_line = "-" * 80
    
    # load target data (raw)
    target_data, _ = hierarchical_dataset(args.target_data, args, mode = "raw")

    select_data = list(np.load(args.select_data))
    
    # setup Domain Discriminator (DD)
    discrimination = DomainStratifying(args, select_data)

    # load target data
    target_data_adjust_raw = Subset(target_data, select_data)
    target_data_adjust = Pseudolabel_Dataset(target_data_adjust_raw, np.full(len(target_data_adjust_raw), 1))

    print(dashed_line)

    # load source data
    source_data_raw, _ = hierarchical_dataset(args.source_data, args, mode = "raw")
    source_data = Pseudolabel_Dataset(source_data_raw, np.full(len(source_data_raw), 0)) 

    del target_data_adjust_raw, source_data_raw

    print(dashed_line)
    
    # setup model
    model = BaselineClassifier(args)

    # load pretrained model (baseline)
    pretrained_state_dict = torch.load(args.saved_model)
    state_dict = model.state_dict()
    for key in list(state_dict.keys()):
        if (("module." + key) in pretrained_state_dict.keys()):
            state_dict[key] = pretrained_state_dict["module." + key].data
        # else:
        #     print(key)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.train()
    # print(model.state_dict())

    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print(f"Trainable params num: {sum(params_num)}")

    # setup loss (not contain sigmoid function)
    criterion = FocalLoss().to(device)

    # get dataloader
    source_loader = get_dataloader(args, source_data, args.batch_size, shuffle=True)
    target_loader = get_dataloader(args, target_data_adjust, args.batch_size, shuffle=True)

    # set up iter dataloader
    source_loader_iter = iter(source_loader)

    del source_data, target_data_adjust

    # start training
    if (args.infer == False):
        model.train()

        # set up optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # set up scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=args.lr,
                    cycle_momentum=False,
                    div_factor=20,
                    final_div_factor=1000,
                    total_steps=args.epochs * (len(select_data) // args.batch_size + 1),
                )
        
        # train
        train_loss_avg = Averager()

        for epoch in range(args.epochs):

            for batch in tqdm(target_loader):
                
                images_target_tensor, labels_target = batch

                try:
                    images_source_tensor, labels_source = next(source_loader_iter)
                except StopIteration:
                    del source_loader_iter
                    source_loader_iter = iter(source_loader)
                    images_source_tensor, labels_source = next(source_loader_iter)

                images_tensor = torch.cat((images_source_tensor, images_target_tensor), 0)
                labels = labels_source + labels_target
                images = images_tensor.to(device)
                preds = model(images)
                loss = criterion(preds, torch.Tensor(labels).view(-1,1).to(device))

                # optimize
                model.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )   # gradient clipping with 5 (Default)
                optimizer.step()
                train_loss_avg.add(loss)
                scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            valid_log = f"\nEpoch: {epoch + 1}/{args.epochs}: "
            valid_log += f"Train_loss: {train_loss_avg.val():0.5f}, Current_lr: {lr:0.7f}\n"
            print(valid_log)
            train_loss_avg.reset()

            torch.save(
                model.state_dict(),
                f"stratify/{args.method}/discriminator.pth"
            )   
            
        model.eval()
    
    del source_loader_iter, source_loader, target_loader
    
    # Domain Stratifying
    discrimination.stratify_DD(target_data, model)
    
    print(dashed_line)
        

if __name__ == "__main__":
    """ Argument """ 
    parser = argparse.ArgumentParser()
    config = load_config("config/default.yaml")
    parser.set_defaults(**config)
    
    parser.add_argument(
        "--source_data", default="data/train/synth/", help="path to source domain data",
    )
    parser.add_argument(
        "--target_data", default="data/train/real/", help="path to target domain data",
    )
    parser.add_argument(
        "--select_data",
        required=True,
        help="path to select data",
    )
    parser.add_argument(
        "--saved_model",
        required=True,
        help="path to pretrained model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="input batch size",
    )
    parser.add_argument(
        "--batch_size_val", type=int, default=512, help="input batch size val",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="number of epochs to train for",
    )
    """ Adaptation """
    parser.add_argument(
        "--num_subsets",
        type=int,
        required=True,
        help="hyper-parameter n, number of subsets partitioned from target domain data",
    )
    parser.add_argument(
        "--method", default="DD", help="select Domain Stratifying method, DD|HDGE",
    )
    parser.add_argument(
        "--discriminator",
        type=str,
        required=True,
        help="choose discriminator, CRNN|TRBA",
    )
    parser.add_argument(
        "--infer", action="store_true", default=False, help="inference or not",
    )

    args = parser.parse_args()

    if args.discriminator == "CRNN":  # CRNN = NVBC
        args.Transformation = "None"
        args.FeatureExtraction = "VGG"
        args.SequenceModeling = "None"
        args.Prediction = "CTC"

    elif args.discriminator == "TRBA":  # TRBA
        args.Transformation = "TPS"
        args.FeatureExtraction = "ResNet"
        args.SequenceModeling = "None"
        args.Prediction = "None"
    
    """ Seed and GPU setting """
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)

    cudnn.benchmark = True  # it fasten training
    cudnn.deterministic = True

    if sys.platform == "win32":
        args.workers = 0

    args.gpu_name = "_".join(torch.cuda.get_device_name().split())
    if sys.platform == "linux":
        args.CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        args.CUDA_VISIBLE_DEVICES = 0  # for convenience

    command_line_input = " ".join(sys.argv)
    print(
        f"Command line input: CUDA_VISIBLE_DEVICES={args.CUDA_VISIBLE_DEVICES} python {command_line_input}"
    )

    main(args)
