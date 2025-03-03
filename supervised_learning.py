import os
import sys
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn

from utils.averager import Averager
from utils.converter import AttnLabelConverter, CTCLabelConverter

from source.model import Model
from source.dataset import hierarchical_dataset, get_dataloader

from test import validation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(opt):
    dashed_line = "-" * 80
    main_log = ""
    opt_log = dashed_line + "\n"

    """ Dataset preparation """
    # source data
    train_data, train_data_log = hierarchical_dataset(opt.train_data, opt)
    if opt.aug:
        train_loader = get_dataloader(opt, train_data, opt.batch_size, shuffle = True, mode = "supervised")
    else:
        train_loader = get_dataloader(opt, train_data, opt.batch_size, shuffle = True)
    
    opt_log += train_data_log

    # validation data
    valid_data, valid_data_log = hierarchical_dataset(opt.valid_data, opt)
    valid_loader = get_dataloader(opt, valid_data, opt.batch_size_val, shuffle = False) # 'True' to check training progress with validation function.
    
    opt_log += valid_data_log

    del train_data, valid_data, train_data_log, valid_data_log

    """ Model configuration """
    if opt.Prediction == "CTC":
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
        opt.sos_token_index = converter.dict["[SOS]"]
        opt.eos_token_index = converter.dict["[EOS]"]
    opt.num_class = len(converter.character)
    
    # setup model
    model = Model(opt)
    opt_log += "Init model\n"

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()

    # load pretrained model
    pretrained = torch.load(opt.saved_model)
    model.load_state_dict(pretrained)
    torch.save(
            pretrained,
            f"./trained_model/{opt.model}_bestmodel.pth"
        )
    opt_log += "Load pretrained model\n"

    del pretrained

    """ Setup loss """
    if opt.Prediction == "CTC":
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
    opt_log += f"Trainable params num: {sum(params_num)}"

    del params_num

    """ Final options """
    opt_log += "------------ Options -------------\n"
    args = vars(opt)
    for k, v in args.items():
        if str(k) == "character" and len(str(v)) > 500:
            opt_log += f"{str(k)}: So many characters to show all: number of characters: {len(str(v))}\n"
        else:
            opt_log += f"{str(k)}: {str(v)}\n"
    opt_log += "---------------------------------------\n"
    print(opt_log)
    main_log += opt_log
    print("Start Training...\n")
    main_log += "Start Training...\n"

    total_iter = (opt.num_epoch * len(train_loader))

    # set up optimizer
    optimizer = torch.optim.AdamW(filtered_parameters, lr=opt.lr, weight_decay = 0.01)

    # set up scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=opt.lr,
                cycle_momentum=False,
                div_factor=20,
                final_div_factor=1000,
                total_steps=total_iter,
            )
    
    train_loss_avg = Averager()
    best_score = float('-inf')
    score_descent = 0
    iteration = 0

    log = ""

    model.train()
    # training loop
    for epoch in tqdm(range(opt.num_epoch)):

        for (images, labels) in tqdm(train_loader):
            batch_size = len(labels)
            
            iteration += 1

            images_tensor = images.to(device)          
            labels_index, labels_length = converter.encode(
                labels, batch_max_length=opt.batch_max_length
            )
            
            if opt.Prediction == "CTC":
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
                model.parameters(), opt.grad_clip
            )   # gradient clipping with 5 (Default)
            optimizer.step()

            train_loss_avg.add(loss)

            scheduler.step()

            if (iteration % opt.val_interval == 0 or iteration == total_iter):
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
                    ) = validation(model, criterion, valid_loader, converter, opt)
                model.train()

                if (current_score >= best_score):
                    score_descent = 0

                    best_score = current_score
                    torch.save(model.state_dict(), f"./trained_model/{opt.model}_bestmodel.pth")
                else:
                    score_descent += 1

                # log
                lr = optimizer.param_groups[0]["lr"]
                valid_log = f'\nValidation at {iteration}/{total_iter}:\n'
                valid_log += f'Train_loss: {train_loss_avg.val():0.3f}, Valid_loss: {valid_loss:0.3f}, '
                valid_log += f'Current_lr: {lr:0.7f}, '
                valid_log += f'Current_score: {current_score:0.2f}, Best_score: {best_score:0.2f}, '
                valid_log += f'Score_descent: {score_descent}\n'
                print(valid_log)

                log += valid_log

                log += "-" * 80 +"\n"

                train_loss_avg.reset()

    # free cache
    torch.cuda.empty_cache()
    
    return
            

if __name__ == "__main__":
    """ Argument """ 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data",
        default="data/train/synth/",
        help="path to source dataset",
    )
    parser.add_argument(
        "--valid_data",
        default="data/val/",
        help="path to validation dataset",
    )
    parser.add_argument(
        "--saved_model",
        required=True, 
        help="path to saved_model to evaluation"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="input batch size")
    parser.add_argument("--batch_size_val", type=int, default=512, help="input batch size val")
    parser.add_argument("--num_epoch", type=int, default=3, help="number of iterations to train for each round")
    parser.add_argument("--val_interval", type=int, default=2000, help="interval between each validation")
    parser.add_argument(
        "--workers", type=int, help="number of data loading workers", default=4
    )
    parser.add_argument(
        "--grad_clip", type=float, default=5, help="gradient clipping value. default=5"
    )
    """ Data Processing """
    parser.add_argument(
        "--batch_max_length", type=int, default=25, help="maximum-label-length"
    )
    parser.add_argument(
        "--imgH", type=int, default=32, help="the height of the input image"
    )
    parser.add_argument(
        "--imgW", type=int, default=100, help="the width of the input image"
    )
    parser.add_argument(
        "--character",
        type=str,
        default="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        help="character label",
    )
    parser.add_argument(
        "--NED", action="store_true", help="for Normalized edit_distance"
    )
    """ Model Architecture """
    parser.add_argument("--model", type=str, required=True, help="CRNN|TRBA") 
    parser.add_argument(
        "--num_fiducial",
        type=int,
        default=20,
        help="number of fiducial points of TPS-STN",
    )
    parser.add_argument(
        "--input_channel",
        type=int,
        default=3,
        help="the number of input channel of Feature extractor",
    )
    parser.add_argument(
        "--output_channel",
        type=int,
        default=512,
        help="the number of output channel of Feature extractor",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="the size of the LSTM hidden state"
    )
    """ Optimizer """ 
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate, 0.001 for Adam",
    )
    """ Experiment """ 
    parser.add_argument(
        "--manual_seed", type=int, default=111, help="for random seed setting"
    )
    parser.add_argument("--aug", action='store_true', default=False, help='augmentation or not')

    opt = parser.parse_args()

    opt.use_IMAGENET_norm = False  # for CRNN and TRBA
    
    if opt.model == "CRNN":  # CRNN = NVBC
        opt.Transformation = "None"
        opt.FeatureExtraction = "VGG"
        opt.SequenceModeling = "BiLSTM"
        opt.Prediction = "CTC"

    elif opt.model == "TRBA":  # TRBA
        opt.Transformation = "TPS"
        opt.FeatureExtraction = "ResNet"
        opt.SequenceModeling = "BiLSTM"
        opt.Prediction = "Attn"

    """ Seed and GPU setting """   
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

    cudnn.benchmark = True  # it fasten training
    cudnn.deterministic = True

    if sys.platform == "win32":
        opt.workers = 0

    opt.gpu_name = "_".join(torch.cuda.get_device_name().split())
    if sys.platform == "linux":
        opt.CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        opt.CUDA_VISIBLE_DEVICES = 0  # for convenience

    command_line_input = " ".join(sys.argv)
    print(
        f"Command line input: CUDA_VISIBLE_DEVICES={opt.CUDA_VISIBLE_DEVICES} python {command_line_input}"
    )

    main(opt)
