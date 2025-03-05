import os
import sys
import random
import argparse
from tqdm import tqdm

import numpy as np
from PIL import ImageFile

import torch
import torch.backends.cudnn as cudnn

from utils.averager import Averager
from utils.converter import AttnLabelConverter, CTCLabelConverter
from utils.load_config import load_config

from source.model import Model
from source.dataset import hierarchical_dataset, get_dataloader

from test import validation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy("file_system")


def main(args):
    dashed_line = "-" * 80
    main_log = ""
    args_log = dashed_line + "\n"
    
    """ Dataset preparation """
    # source domain data
    print(dashed_line)
    print("Load training data (source domain)...")
    train_data, train_data_log = hierarchical_dataset(args.train_data, args)
    if args.aug:
        train_loader = get_dataloader(args, train_data, args.batch_size, shuffle=True, mode="supervised")
    else:
        train_loader = get_dataloader(args, train_data, args.batch_size, shuffle=True)
    
    args_log += train_data_log

    # validation data
    print(dashed_line)
    print("Load validation data...")
    valid_data, valid_data_log = hierarchical_dataset(args.valid_data, args)
    valid_loader = get_dataloader(args, valid_data, args.batch_size_val, shuffle=False) # "True" to check training progress with validation function.
    
    args_log += valid_data_log

    del train_data, valid_data, train_data_log, valid_data_log
    
    print(dashed_line)
    print("Init model")
    args_log += "Init model\n"

    """ Model configuration """
    if args.Prediction == "CTC":
        converter = CTCLabelConverter(args.character)
    else:
        converter = AttnLabelConverter(args.character)
        args.sos_token_index = converter.dict["[SOS]"]
        args.eos_token_index = converter.dict["[EOS]"]
    args.num_class = len(converter.character)
    
    # setup model
    model = Model(args)

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()

    # load pretrained model
    if args.saved_model != "":
        pretrained = torch.load(args.saved_model)
        model.load_state_dict(pretrained)
        torch.save(
            pretrained,
            f"./trained_model/{args.model}_supervised.pth",
        )
        print(f"Load pretrained model from {args.saved_model}")
        args_log += "Load pretrained model\n"

        del pretrained

    """ Setup loss """
    if args.Prediction == "CTC":
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        # ignore [PAD] token
        criterion = torch.nn.CrossEntropyLoss(ignore_index=converter.dict["[PAD]"]).to(device)
    
    # filter that only require gradient descent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print(f"Trainable params num: {sum(params_num)}")
    args_log += f"Trainable params num: {sum(params_num)}"

    del params_num

    """ Final options """
    print("------------ Options -------------")
    args_log += "------------ Options -------------\n"
    opt = vars(args)
    for k, v in opt.items():
        if str(k) == "character" and len(str(v)) > 500:
            print(f"{str(k)}: So many characters to show all: number of characters: {len(str(v))}")
            args_log += f"{str(k)}: So many characters to show all: number of characters: {len(str(v))}\n"
        else:
            print(f"{str(k)}: {str(v)}")
            args_log += f"{str(k)}: {str(v)}\n"
    print(dashed_line)
    args_log += "---------------------------------------\n"
    main_log += args_log
    print("Start Supervised Learning (Scene Text Recognition - STR)...\n")
    main_log += "Start Supervised Learning (Scene Text Recognition - STR)...\n"

    total_iter = (args.epochs * len(train_loader))

    # set up optimizer
    optimizer = torch.optim.AdamW(filtered_parameters, lr=args.lr, weight_decay=args.weight_decay)

    # set up scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.lr,
                cycle_momentum=False,
                div_factor=20,
                final_div_factor=1000,
                total_steps=total_iter,
            )
    
    train_loss_avg = Averager()
    best_score = float("-inf")
    score_descent = 0

    log = ""

    # training loop
    for epoch in tqdm(range(args.epochs)):

        # training part
        model.train()
        for (images, labels) in tqdm(train_loader):
            batch_size = len(labels)
            
            images_tensor = images.to(device)          
            labels_index, labels_length = converter.encode(
                labels, batch_max_length=args.batch_max_length
            )
            
            if args.Prediction == "CTC":
                preds = model(images_tensor)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds_log_softmax = preds.log_softmax(2).permute(1, 0, 2)
                loss = criterion(preds_log_softmax, labels_index, preds_size, labels_length)
            else:
                preds = model(images_tensor, labels_index[:, :-1])  # align with Attention.forward
                target = labels_index[:, 1:]  # without [SOS] Symbol
                loss = criterion(
                    preds.view(-1, preds.shape[-1]), target.contiguous().view(-1)
                )

            model.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )   # gradient clipping with 5 (Default)
            optimizer.step()

            train_loss_avg.add(loss)

            scheduler.step()

        # valiation part
        model.eval()
        with torch.no_grad():
            (
                valid_loss,
                current_score,
                preds,
                confidence_score,
                labels,
                infer_time,
                length_of_data,
            ) = validation(model, criterion, valid_loader, converter, args)
        model.train()

        if (current_score >= best_score):
            score_descent = 0

            best_score = current_score
            torch.save(
                model.state_dict(),
                f"./trained_model/{args.model}_supervised.pth",
            )
        else:
            score_descent += 1

        # log
        lr = optimizer.param_groups[0]["lr"]
        valid_log = f"\n\nEpoch {epoch + 1}/{args.epochs}:\n"
        valid_log += f"Train_loss: {train_loss_avg.val():0.3f}, Valid_loss: {valid_loss:0.3f}, "
        valid_log += f"Current_lr: {lr:0.7f},\n"
        valid_log += f"Current_score: {current_score:0.2f}, Best_score: {best_score:0.2f}, "
        valid_log += f"Score_descent: {score_descent}\n"
        print(valid_log)

        log += valid_log

        log += "-" * 80 +"\n"

        train_loss_avg.reset()
    
    main_log += log

    # free cache
    torch.cuda.empty_cache()
    
    # save log
    print(main_log, file= open(f"log/supervised_learning.txt", "w"))
    
    return
            

if __name__ == "__main__":
    """ Argument """ 
    parser = argparse.ArgumentParser()
    config = load_config("config/STR.yaml")
    parser.set_defaults(**config)
    
    parser.add_argument(
        "--train_data", default="data/train/synth/", help="path to training dataset",
    )
    parser.add_argument(
        "--valid_data", default="data/val/", help="path to validation dataset",
    )
    parser.add_argument(
        "--saved_model", default="", help="path to pretrained model (to continue training)",
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
    parser.add_argument(
        "--val_interval", type=int, default=1000, help="interval between each validation",
    )
    parser.add_argument(
        "--NED", action="store_true", help="for Normalized edit_distance",
    )
    """ Model Architecture """
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="CRNN|TRBA",
    )
    """ Training """ 
    parser.add_argument(
        "--aug", action="store_true", default=False, help="augmentation or not",
    )

    args = parser.parse_args()
    
    if args.model == "CRNN":  # CRNN = NVBC
        args.Transformation = "None"
        args.FeatureExtraction = "VGG"
        args.SequenceModeling = "BiLSTM"
        args.Prediction = "CTC"

    elif args.model == "TRBA":  # TRBA
        args.Transformation = "TPS"
        args.FeatureExtraction = "ResNet"
        args.SequenceModeling = "BiLSTM"
        args.Prediction = "Attn"

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
