import os
import sys
import time
import random
import argparse
from tqdm import tqdm

import numpy as np
from PIL import ImageFile

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset

from utils.averager import Averager
from utils.converter import AttnLabelConverter, CTCLabelConverter
from utils.load_config import load_config

from source.model import Model
from source.dataset import Pseudolabel_Dataset, hierarchical_dataset, get_dataloader

from test import validation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy("file_system")


def pseudo_labeling(args, model, converter, target_data, adapting_list, round):
    """ Make prediction and return them """

    # get adapt_data
    data = Subset(target_data, adapting_list)
    data = Pseudolabel_Dataset(data, adapting_list)
    dataloader = get_dataloader(args, data, args.batch_size_val, shuffle=False)
    
    model.eval()
    with torch.no_grad():
        list_adapt_data = list()
        list_pseudo_data = list()
        list_pseudo_label = list()

        mean_conf = 0

        for (image_tensors, image_indexs) in tqdm(dataloader):
            batch_size = len(image_indexs)
            image = image_tensors.to(device)

            if args.Prediction == "CTC":
                preds = model(image)
            else:
                text_for_pred = (
                        torch.LongTensor(batch_size)
                        .fill_(args.sos_token_index)
                        .to(device)
                    )
                preds = model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, preds_size)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            for pred, pred_max_prob, index in zip(
                preds_str, preds_max_prob, image_indexs
            ):
                if args.Prediction == "Attn":
                    pred_EOS = pred.find("[EOS]")
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]
                
                if ( 
                    "[PAD]" in pred 
                    or "[UNK]" in pred 
                    or "[SOS]" in pred 
                ):
                    list_pseudo_label.append(pred)
                    continue
                
                # calculate confidence score (= multiply of pred_max_prob)
                if len(pred_max_prob.cumprod(dim=0)) > 0:
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1].item()
                else:
                    list_pseudo_label.append(pred)
                    continue

                list_adapt_data.append(index)
                list_pseudo_data.append(pred)

                mean_conf += confidence_score

    mean_conf /= (len(list_adapt_data))
    # adjust mean_conf (round_down)
    mean_conf = int(mean_conf * 10) / 10
    
    # save pseudo-labels
    with open(f"stratify/{args.method}/pseudolabel_{round}.txt", "w") as file:
        for string in list_pseudo_label:
            file.write(string + "\n")

    # free cache
    torch.cuda.empty_cache()
                
    return list_adapt_data, list_pseudo_data, mean_conf

           
def self_training(args, filtered_parameters, model, criterion, converter, \
                  source_loader, valid_loader, adapting_loader, mean_conf, round = 0):

    num_iter = (args.total_iter // args.val_interval) // args.num_subsets * args.val_interval

    if round == 1:
        num_iter += (args.total_iter // args.val_interval) % args.num_subsets * args.val_interval

    # set up iter dataloader
    source_loader_iter = iter(source_loader)
    adapting_loader_iter = iter(adapting_loader)

    # set up optimizer
    optimizer = torch.optim.AdamW(filtered_parameters, lr=args.lr, weight_decay=args.weight_decay)

    # set up scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.lr,
                cycle_momentum=False,
                div_factor=20,
                final_div_factor=1000,
                total_steps=num_iter,
            )
    
    train_loss_avg = Averager()
    source_loss_avg = Averager()
    adapting_loss_avg = Averager()
    best_score = float("-inf")
    score_descent = 0

    log = ""

    model.train()
    # training loop
    for iteration in tqdm(
        range(0, num_iter + 1),
        total=num_iter,
        position=0,
        leave=True,
    ):
        if (iteration % args.val_interval == 0 or iteration == num_iter):
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
                    f"trained_model/{args.method}/StrDA_round{round}.pth",
                )
            else:
                score_descent += 1

            # log
            lr = optimizer.param_groups[0]["lr"]
            valid_log = f"\nValidation at {iteration}/{num_iter}:\n"
            valid_log += f"Train_loss: {train_loss_avg.val():0.4f}, Valid_loss: {valid_loss:0.4f}, "
            valid_log += f"Source_loss: {source_loss_avg.val():0.4f}, Adapting_loss: {adapting_loss_avg.val():0.4f},\n"
            valid_log += f"Current_lr: {lr:0.7f}, "
            valid_log += f"Current_score: {current_score:0.2f}, Best_score: {best_score:0.2f}, "
            valid_log += f"Score_descent: {score_descent}\n"
            print(valid_log)

            log += valid_log

            log += "-" * 80 +"\n"

            train_loss_avg.reset()
            source_loss_avg.reset()
            adapting_loss_avg.reset()

        if iteration == num_iter:
            log += f"Stop training at iteration: {iteration}!\n"
            print(f"Stop training at iteration: {iteration}!\n")
            break

        # training part
        """ loss of source domain """
        try:
            images_source_tensor, labels_source = next(source_loader_iter)
        except StopIteration:
            del source_loader_iter
            source_loader_iter = iter(source_loader)
            images_source_tensor, labels_source = next(source_loader_iter)

        images_source = images_source_tensor.to(device)          
        labels_source_index, labels_source_length = converter.encode(
            labels_source, batch_max_length=args.batch_max_length
        )

        batch_source_size = len(labels_source)
        if args.Prediction == "CTC":
            preds_source = model(images_source)
            preds_source_size = torch.IntTensor([preds_source.size(1)] * batch_source_size)
            preds_source_log_softmax = preds_source.log_softmax(2).permute(1, 0, 2)
            loss_source = criterion(preds_source_log_softmax, labels_source_index, preds_source_size, labels_source_length)
        else:
            preds_source = model(images_source, labels_source_index[:, :-1])  # align with Attention.forward
            target_source = labels_source_index[:, 1:]  # without [SOS] Symbol
            loss_source = criterion(
                preds_source.view(-1, preds_source.shape[-1]), target_source.contiguous().view(-1)
            )

        """ loss of semi """
        try:
            images_unlabel_tensor, labels_adapting = next(adapting_loader_iter)
        except StopIteration:
            del adapting_loader_iter
            adapting_loader_iter = iter(adapting_loader)
            images_unlabel_tensor, labels_adapting = next(adapting_loader_iter)
        
        images_unlabel = images_unlabel_tensor.to(device)
        labels_adapting_index, labels_adapting_length = converter.encode(
            labels_adapting, batch_max_length=args.batch_max_length
        )

        batch_unlabel_size = len(labels_adapting)
        if args.Prediction == "CTC":
            preds_adapting = model(images_unlabel)
            preds_adapting_size = torch.IntTensor([preds_adapting.size(1)] * batch_unlabel_size)
            preds_adapting_log_softmax = preds_adapting.log_softmax(2).permute(1, 0, 2)
            loss_adapting = criterion(preds_adapting_log_softmax, labels_adapting_index, preds_adapting_size, labels_adapting_length)
        else:
            preds_adapting = model(images_unlabel, labels_adapting_index[:, :-1])  # align with Attention.forward
            target_adapting = labels_adapting_index[:, 1:]  # without [SOS] Symbol
            loss_adapting = criterion(
                preds_adapting.view(-1, preds_adapting.shape[-1]), target_adapting.contiguous().view(-1)
            )

        loss = (1 - mean_conf) * loss_source + loss_adapting * mean_conf

        model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip
        )   # gradient clipping with 5 (Default)
        optimizer.step()

        train_loss_avg.add(loss)
        source_loss_avg.add(loss_source)
        adapting_loss_avg.add(loss_adapting)

        scheduler.step()

    model.eval()

    # save model
    # torch.save(
    #     model.state_dict(),
    #     f"trained_model/{args.method}/StrDA_round{round}.pth",
    # )

    # save log
    print(log, file= open(f"log/{args.method}/log_self_training_round{round}.txt", "w"))

    # free cache
    torch.cuda.empty_cache()


def main(args):
    dashed_line = "-" * 80
    main_log = ""
    
    if args.method == "HDGE":
        if args.beta == -1:
            raise ValueError("Please set beta value for HDGE method.")
        relative_path = f"{args.method}/{args.beta}_beta/{args.num_subsets}_subsets"
    else:
        if args.discriminator == "":
            raise ValueError("Please set discriminator for DD method.")
        relative_path = f"{args.method}/{args.discriminator}/{args.num_subsets}_subsets"
    
    # to make directories for saving models and logs if not exist
    os.makedirs(f"log/{relative_path}/", exist_ok=True)
    os.makedirs(f"trained_model/{relative_path}/", exist_ok=True)

    # load source domain data
    print(dashed_line)
    main_log = dashed_line + "\n"
    print("Load source domain data...")
    main_log += "Load source domain data...\n"
    
    source_data, source_data_log = hierarchical_dataset(args.source_data, args)
    source_loader = get_dataloader(args, source_data, args.batch_size, shuffle=True, aug=args.aug)

    print(source_data_log, end="")
    main_log += source_data_log
    
    # load target domain data (raw)
    print(dashed_line)
    main_log += dashed_line + "\n"
    print("Load target domain data...")
    main_log += "Load target domain data...\n"
    
    target_data,  target_data_log= hierarchical_dataset(args.target_data, args, mode="raw")

    print(target_data_log, end="")
    main_log += target_data_log

    # load validation data
    print(dashed_line)
    main_log += dashed_line + "\n"
    print("Load validation data...")
    main_log += "Load validation data...\n"
    
    valid_data, valid_data_log = hierarchical_dataset(args.valid_data, args)
    valid_loader = get_dataloader(args, valid_data, args.batch_size_val, shuffle=False) # "True" to check training progress with validation function.
    
    print(valid_data_log, end="")
    main_log += valid_data_log

    """ Model configuration """
    if args.Prediction == "CTC":
        converter = CTCLabelConverter(args.character)
    else:
        converter = AttnLabelConverter(args.character)
        args.sos_token_index = converter.dict["[SOS]"]
        args.eos_token_index = converter.dict["[EOS]"]
    args.num_class = len(converter.character)
    
    # setup model
    print(dashed_line)
    main_log += dashed_line + "\n"
    print("Init model")
    main_log += "Init model\n"
    model = Model(args)

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()

    # load pretrained model
    try:
        pretrained = torch.load(args.saved_model)
        model.load_state_dict(pretrained)
    except:
        raise ValueError("The pre-trained weights do not match the model! Carefully check!")
    
    torch.save(
        pretrained,
        f"trained_model/{relative_path}/{args.model}_round0.pth"
    )
    print(f"Load pretrained model from {args.saved_model}")
    main_log += f"Load pretrained model from {args.saved_model}\n"
    
    # setup loss
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
    main_log += f"Trainable params num: {sum(params_num)}\n"

    """ Final options """
    print("------------ Options -------------")
    main_log += "------------ Options -------------\n"
    opt = vars(args)
    for k, v in opt.items():
        if str(k) == "character" and len(str(v)) > 500:
            print(f"{str(k)}: So many characters to show all: number of characters: {len(str(v))}")
            main_log += f"{str(k)}: So many characters to show all: number of characters: {len(str(v))}\n"
        else:
            print(f"{str(k)}: {str(v)}")
            main_log += f"{str(k)}: {str(v)}\n"
    print(dashed_line)
    main_log += dashed_line + "\n"
    print("Start Adapting (Scene Text Recognition - STR)...\n")
    main_log += "Start Adapting (Scene Text Recognition - STR)...\n"
    
    for round in range(args.num_subsets):

        print(f"Round {round+1}/{args.num_subsets}: \n")
        main_log += f"\nRound {round+1}/{args.num_subsets}: \n" 

        # load best model of previous round
        print(f"- Load best model of round {round}.")
        main_log +=  f"- Load best model of round {round}. \n"
        model.load_state_dict(
            torch.load(f"trained_model/{relative_path}/{args.model}_round{round}.pth")             
        )

        # select subset
        try:
            adapting_list = list(np.load(f"stratify/{relative_path}/subset_{round + 1}.npy"))
        except:
            raise ValueError(f"stratify/{relative_path}/subset_{round + 1}.npy not found.")
        
        # assign pseudo labels
        print("- Pseudo labeling...\n")
        main_log += "- Pseudo labeling...\n"
        list_adapt_data, list_pseudo_data, mean_conf = pseudo_labeling(
                args, model, converter, target_data, adapting_list, round + 1
            )

        print(f"- Number of adapting data: {len(list_adapt_data)}")
        main_log += f"- Number of adapting data: {len(list_adapt_data)} \n"
        print(f"- Mean of confidence score: {mean_conf}")
        main_log += f"- Mean of confidence scores: {mean_conf} \n"
    
        # restrict adapting data
        adapting_data = Subset(target_data, list_adapt_data)
        adapting_data = Pseudolabel_Dataset(adapting_data, list_pseudo_data)

        # get dataloader
        adapting_loader = get_dataloader(args, adapting_data, args.batch_size, shuffle=True, aug=args.aug)

        # self-training
        print(dashed_line)
        print("- Seft-training...")
        main_log += "\n- Seft-training"

        # adjust mean_conf (round_down)
        mean_conf = int(mean_conf * 10)

        self_training_start = time.time()
        if (round >= args.checkpoint):
            self_training(args, filtered_parameters, model, criterion, converter, \
                        source_loader, valid_loader, adapting_loader, mean_conf, round + 1)
        self_training_end = time.time()

        print(f"Processing time: {self_training_end - self_training_start}s")
        print(f"Saved log for adapting round to: 'log/{args.method}/log_self_training_round{round + 1}.txt'")
        adapt_log += f"\nProcessing time: {self_training_end - self_training_start}s"
        adapt_log += f"\nSaved log for adapting round to: 'log/{args.method}/log_self_training_round{round + 1}.txt'"

        main_log += "\n" + dashed_line + "\n"

        print(dashed_line)
        print(dashed_line)
        print(dashed_line)
    
    # free cache
    torch.cuda.empty_cache()
    
    # save log
    print(main_log, file= open(f"log/{args.method}/log_StrDA.txt", "w"))
    
    return            

if __name__ == "__main__":
    """ Argument """
    parser = argparse.ArgumentParser()
    config = load_config("config/STR.yaml")
    parser.set_defaults(**config)

    parser.add_argument(
        "--source_data", default="data/train/synth/", help="path to source dataset",
    )
    parser.add_argument(
        "--target_data", default="data/train/real/", help="path to adaptation dataset",
    )
    parser.add_argument(
        "--valid_data", default="data/val/", help="path to validation dataset",
    )
    parser.add_argument(
        "--saved_model",
        required=True, 
        help="path to source-trained model for adaptation",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="input batch size",
    )
    parser.add_argument(
        "--batch_size_val", type=int, default=512, help="input batch size val",
    )
    parser.add_argument(
        "--total_iter", type=int, default=50000, help="number of iterations to train for",
    )
    parser.add_argument(
        "--val_interval", type=int, default=500, help="interval between each validation",
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
    """ Adaptation """
    parser.add_argument(
        "--num_subsets",
        type=int,
        required=True,
        help="hyper-parameter n, number of subsets partitioned from target domain data",
    )
    parser.add_argument(
        "--method",
        required=True,
        help="select Domain Stratifying method, DD|HDGE",
    )
    parser.add_argument("--discriminator", default="", help="for DD method, choose discriminator, CRNN|TRBA")
    parser.add_argument("--beta", type=float, default=-1, help="for HDGE method, hyper-parameter beta, 0<beta<1")
    parser.add_argument(
        "--aug", action="store_true", default=False, help="augmentation or not",
    )
    parser.add_argument(
        "--checkpoint", type=int, default=0, help="iteration of checkpoint",
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
