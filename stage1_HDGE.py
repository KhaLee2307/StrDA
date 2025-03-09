import os
import sys
import random
import argparse

import numpy as np
from PIL import ImageFile

import torch
import torch.backends.cudnn as cudnn

import utils.utils_HDGE as utils
from utils.load_config import load_config

from modules.discriminators import define_Dis

import source.HDGE as md
from source.stratify import DomainStratifying
from source.dataset import hierarchical_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy("file_system")
    
    
def main(args):
    dashed_line = "-" * 80
    
    # to make directories for saving results and trained models
    args.saved_path = f"stratify/{args.method}/{args.beta}_beta"
    os.makedirs(f"{args.saved_path}/{args.num_subsets}_subsets/", exist_ok=True)
    
    str_ids = args.gpu_ids.split(",")
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    # print(not args.no_dropout)
    
    # training part
    if args.train:
        print(dashed_line)
        model = md.HDGE(args)
        model.train(args)
    
    # inference part
    print(dashed_line)
    print("Start Inference")

    # load target domain data (raw)
    print("Load target domain data for inference...")
    target_data_raw, target_data_log = hierarchical_dataset(args.target_data, args, mode = "raw")
    print(target_data_log, end="")
    
    try:
        select_data = list(np.load(args.select_data))
    except:
        print("\n [*][WARNING] NO available select_data!")
        print(" [*][WARNING] You are using all target domain data!\n")
        select_data = list(range(len(target_data_raw)))
    
    print(dashed_line)

    dis_source = define_Dis(input_nc=3, ndf=args.ndf, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)
    dis_target = define_Dis(input_nc=3, ndf=args.ndf, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)

    utils.print_networks([dis_source,dis_target], ["Da","Db"])

    try:
        ckpt = utils.load_checkpoint("%s/HDGE_gen_dis.ckpt" % (args.checkpoint_dir))
        dis_source.load_state_dict(ckpt["Da"])
        dis_target.load_state_dict(ckpt["Db"])
        
        print(dashed_line)
        # Domain Stratifying (Harmonic Domain Gap Estimator - HDGE)
        HDGE = DomainStratifying(args, select_data)
        HDGE.stratify_HDGE(target_data_raw, dis_source, dis_target, args.beta)
        
        print("\nAll information is saved in " + f"{args.saved_path}/")
        print("The trained weights are saved at " + f"{args.checkpoint_dir}/HDGE_gen_dis.ckpt")
    
    except:
        print("\n [*][WARNING] STOP Domain Stratifying!")
        print(" [*][WARNING] NO checkpoint!")
        print(" [*][WARNING] Please train the model first!")
        print(" [*][WARNING] Please check the checkpoint directory!\n")
        raise ValueError("NO checkpoint!")
    
    print(dashed_line)
    return


if __name__ == "__main__":
    """ Argument """
    parser = argparse.ArgumentParser()
    config = load_config("config/HDGE.yaml")
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
        "--checkpoint_dir", type=str, default="stratify/HDGE", help="models are saved here",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="input batch size",
    )
    parser.add_argument(
        "--batch_size_val", type=int, default=128, help="input batch size val",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="number of epochs to train for",
    )
    parser.add_argument(
        "--no_dropout", action="store_true", help="no dropout for the generator",
    )
    parser.add_argument(
        "--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0  0,1,2, 0,2",
    )
    """ Adaptation """
    parser.add_argument(
        "--num_subsets",
        type=int,
        required=True,
        help="hyper-parameter n, number of subsets partitioned from target domain data",
    )
    parser.add_argument(
        "--beta",
        type=float,
        required=True,
        help="hyper-parameter beta in HDGE formula, 0<beta<1",
    )
    parser.add_argument(
        "--train", action="store_true", default=False, help="training or not",
    )

    args = parser.parse_args()
    
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
