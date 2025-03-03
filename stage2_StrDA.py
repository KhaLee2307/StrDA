import os
import sys
import time
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset

from utils.averager import Averager
from utils.converter import AttnLabelConverter, CTCLabelConverter

from source.model import Model
from source.dataset import Pseudolabel_Dataset, hierarchical_dataset, get_dataloader

from test import validation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pseudo_labeling(opt, model, converter, adapt_data, adapt_list, round):
    """ Make prediction and return them """

    # get adapt_data
    data = Subset(adapt_data, adapt_list)
    data = Pseudolabel_Dataset(data, adapt_list)
    dataloader = get_dataloader(opt, data, opt.batch_size_val, shuffle=False)
    
    model.eval()
    with torch.no_grad():
        list_adapt_data = list()
        pseudo_adapt = list()

        mean_conf = 0

        for (image_tensors, image_indexs) in tqdm(dataloader):
            batch_size = len(image_indexs)
            image = image_tensors.to(device)

            if opt.Prediction == "CTC":
                preds = model(image)
            else:
                text_for_pred = (
                        torch.LongTensor(batch_size)
                        .fill_(opt.sos_token_index)
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
                if opt.Prediction == "Attn":
                    pred_EOS = pred.find("[EOS]")
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]
                
                # calculate confidence score (= multiply of pred_max_prob)
                if len(pred_max_prob.cumprod(dim=0)) > 0:
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1].item()
                else:
                    confidence_score = 0

                if ( 
                    "[PAD]" in pred 
                    or "[UNK]" in pred 
                    or "[SOS]" in pred 
                ): 
                    continue

                list_adapt_data.append(index)
                pseudo_adapt.append(pred)

                mean_conf += confidence_score

    mean_conf /= (len(list_adapt_data))
    mean_conf = int(mean_conf * 100) / 100

    del dataloader

    # save pseudo-labels
    with open(f'stratify/{opt.method}/pseudolabel_{round}.txt', "w") as file:
        for string in pseudo_adapt:
            file.write(string + "\n")

    # free cache
    torch.cuda.empty_cache()
                
    return list_adapt_data, pseudo_adapt, mean_conf

           
def self_training(opt, filtered_parameters, model, criterion, converter, \
                  source_loader, valid_loader, adapting_loader, mean_conf, round = 0):

    num_iter = (opt.total_iter // opt.val_interval) // opt.num_subsets * opt.val_interval

    if round == 1:
        num_iter += (opt.total_iter // opt.val_interval) % opt.num_subsets * opt.val_interval

    # set up iter dataloader
    source_loader_iter = iter(source_loader)
    adapting_loader_iter = iter(adapting_loader)

    # set up optimizer
    optimizer = torch.optim.AdamW(filtered_parameters, lr=opt.lr, weight_decay = 0.01)

    # set up scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=opt.lr,
                cycle_momentum=False,
                div_factor=20,
                final_div_factor=1000,
                total_steps=num_iter,
            )
    
    train_loss_avg = Averager()
    source_loss_avg = Averager()
    adapting_loss_avg = Averager()
    best_score = float('-inf')
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
        if (iteration % opt.val_interval == 0 or iteration == num_iter):
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
                torch.save(model.state_dict(), f"./trained_model/{opt.method}/StrDA_round{round}.pth")
            else:
                score_descent += 1

            # log
            lr = optimizer.param_groups[0]["lr"]
            valid_log = f'\nValidation at {iteration}/{num_iter}:\n'
            valid_log += f'Train_loss: {train_loss_avg.val():0.4f}, Valid_loss: {valid_loss:0.4f}, '
            valid_log += f'Source_loss: {source_loss_avg.val():0.4f}, Adapting_loss: {adapting_loss_avg.val():0.4f},\n'
            valid_log += f'Current_lr: {lr:0.7f}, '
            valid_log += f'Current_score: {current_score:0.2f}, Best_score: {best_score:0.2f}, '
            valid_log += f'Score_descent: {score_descent}\n'
            print(valid_log)

            log += valid_log

            log += "-" * 80 +"\n"

            train_loss_avg.reset()
            source_loss_avg.reset()
            adapting_loss_avg.reset()

        if iteration == num_iter:
            log += f'Stop training at iteration: {iteration}!\n'
            print(f'Stop training at iteration: {iteration}!\n')
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
            labels_source, batch_max_length=opt.batch_max_length
        )

        batch_source_size = len(labels_source)
        if opt.Prediction == "CTC":
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
            labels_adapting, batch_max_length=opt.batch_max_length
        )

        batch_unlabel_size = len(labels_adapting)
        if opt.Prediction == "CTC":
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

        loss = (10 - mean_conf) * loss_source + loss_adapting * mean_conf

        model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), opt.grad_clip
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
    #     f"./trained_model/{opt.method}/StrDA_round{round}.pth"
    # )

    # save log
    print(log, file= open(f'log/{opt.method}/log_self_training_round{round}.txt', 'w'))

    del optimizer, scheduler, source_loader_iter, adapting_loader_iter, train_loss_avg, source_loss_avg, adapting_loss_avg

    # free cache
    torch.cuda.empty_cache()


def main(opt):
    dashed_line = "-" * 80
    main_log = ""
    opt_log = dashed_line + "\n"

    """ Create folder for log and trained model """
    os.makedirs(f'log/{opt.method}/', exist_ok=True)
    os.makedirs(f'trained_model/{opt.method}/', exist_ok=True)

    """ Dataset preparation """
    # source data
    source_data, source_data_log = hierarchical_dataset(opt.source_data, opt)
    if opt.aug:
        source_loader = get_dataloader(opt, source_data, opt.batch_size, shuffle = True, mode = "adapt")
    else:
        source_loader = get_dataloader(opt, source_data, opt.batch_size, shuffle = True)
    
    opt_log += source_data_log

    # validation data
    valid_data, valid_data_log = hierarchical_dataset(opt.valid_data, opt)
    valid_loader = get_dataloader(opt, valid_data, opt.batch_size_val, shuffle = False) # 'True' to check training progress with validation function.
    
    opt_log += valid_data_log

    # adaptation data
    adapt_data,  adapt_data_log= hierarchical_dataset(opt.adapt_data, opt, mode = "raw")

    opt_log += adapt_data_log

    del source_data, valid_data, source_data_log, valid_data_log, adapt_data_log

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
            f"./trained_model/{opt.method}/StrDA_round0.pth"
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
    print("Start Adapting...\n")
    main_log += "Start Adapting...\n"
    
    for round in range(opt.num_subsets):

        adapt_log = ""
        print(f"\nRound {round+1}/{opt.num_subsets}: \n")
        adapt_log += f"\nRound {round+1}/{opt.num_subsets}: \n"

        # load best model of previous round
        adapt_log +=  f"- Load best model of previous round ({round}). \n"
        pretrained = torch.load(f"./trained_model/{opt.method}/StrDA_round{round}.pth")
        model.load_state_dict(pretrained)
        del pretrained

        # select subset
        adapting_list = list(np.load(f'stratify/{opt.method}/subset_{round + 1}.npy'))

        # assign pseudo labels
        print("- Pseudo labeling \n")
        adapt_log += "- Pseudo labeling \n"
        list_adapt_data, pseudo_adapt, mean_conf = pseudo_labeling(
                opt, model, converter, adapt_data, adapting_list, round + 1
            )

        data_log = ""
        data_log += f"-- Number of apating data: {len(list_adapt_data)} \n"
        data_log += f"-- Mean of confidences: {mean_conf} \n"

        print(data_log)
        adapt_log += data_log

        # restrict adapting data
        adapting_data = Subset(adapt_data, list_adapt_data)
        adapting_data = Pseudolabel_Dataset(adapting_data, pseudo_adapt)

        if opt.aug == True:
            adapting_loader = get_dataloader(opt, adapting_data, opt.batch_size, shuffle=True, mode="adapt")
        else:
            adapting_loader = get_dataloader(opt, adapting_data, opt.batch_size, shuffle=True)

        del adapting_list, adapting_data, list_adapt_data, pseudo_adapt

        # self-training
        print(dashed_line)
        print("- Seft-training...")
        adapt_log += "\n- Seft-training"

        # adjust mean_conf (round_down)
        mean_conf = int(mean_conf * 10)

        self_training_start = time.time()
        if (round >= opt.checkpoint):
            self_training(opt, filtered_parameters, model, criterion, converter, \
                        source_loader, valid_loader, adapting_loader, mean_conf, round + 1)
        self_training_end = time.time()

        print(f"Processing time: {self_training_end - self_training_start}s")
        print(f"Saved log for adapting round to: 'log/{opt.method}/log_self_training_round{round + 1}.txt'")
        adapt_log += f"\nProcessing time: {self_training_end - self_training_start}s"
        adapt_log += f"\nSaved log for adapting round to: 'log/{opt.method}/log_self_training_round{round + 1}.txt'"

        adapt_log += "\n" + dashed_line + "\n"
        main_log += adapt_log

        print(dashed_line)
        print(dashed_line)
        print(dashed_line)
    
    # save log
    print(main_log, file= open(f'log/{opt.method}/log_StrDA.txt', 'w'))
    return
            

if __name__ == "__main__":
    """ Argument """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_data",
        default="data/train/synth/",
        help="path to source dataset",
    )
    parser.add_argument(
        "--valid_data",
        default="data/val/",
        help="path to validation dataset",
    )
    parser.add_argument(
        "--adapt_data",
        default="data/train/real/",
        help="path to adaptation dataset",
    )
    parser.add_argument(
        "--saved_model",
        required=True, 
        help="path to saved_model to evaluation"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="input batch size")
    parser.add_argument("--batch_size_val", type=int, default=512, help="input batch size val")
    parser.add_argument("--total_iter", type=int, default=50000, help="number of iterations to train for each round")
    parser.add_argument("--val_interval", type=int, default=500, help="interval between each validation")
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
    """ Adaptation """
    parser.add_argument("--num_subsets", type=int, required = True, help="hyper-parameter n, number of subsets partitioned from target domain data")
    parser.add_argument("--method", required = True, help="select Domain Stratifying method, DD|HDGE")
    parser.add_argument("--aug", action='store_true', default=False, help='augmentation or not')
    parser.add_argument("--checkpoint", type=int, default=0, help="iteration of checkpoint")

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
