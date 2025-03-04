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
    str_ids = args.gpu_ids.split(",")
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    print(not args.no_dropout)
    
    if args.train:
        print("Training")
        model = md.HDGE(args)
        model.train(args)
    if args.infer:
        print("Inference")

        # load target data (raw)
        target_data, _ = hierarchical_dataset(args.target_data, args, mode = "raw")

        # select_data = list(np.load(args.select_data))
        select_data = list(range(len(target_data)))

        # setup Harmonic Domain Gap Estimator (HDGE)
        hdge = DomainStratifying(args, select_data)

        dis_source = define_Dis(input_nc=3, ndf=args.ndf, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)
        dis_target = define_Dis(input_nc=3, ndf=args.ndf, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)

        utils.print_networks([dis_source,dis_target], ["Da","Db"])

        try:
            ckpt = utils.load_checkpoint("%s/HDGE_gen_dis.ckpt" % (args.checkpoint_dir))
            dis_source.load_state_dict(ckpt["Da"])
            dis_target.load_state_dict(ckpt["Db"])
        except:
            print(" [*] No checkpoint!")

        # Domain Stratifying
        hdge.stratify_HDGE(target_data, dis_source, dis_target, args.beta)


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
        "--batch_size", type=int, default=16, help="input batch size",
    )
    parser.add_argument(
        "--batch_size_val", type=int, default=128, help="input batch size val",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="number of epochs to train for",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="stratify/HDGE/", help="models are saved here",
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
        "--method", default="HDGE", help="select Domain Stratifying method, DD|HDGE",
    )
    parser.add_argument(
        "--beta",
        type=float,
        required=True,
        help="hyper-parameter beta in HDGE formula",
    )
    parser.add_argument(
        "--train", action="store_true", default=False, help="training or not",
    )
    parser.add_argument(
        "--infer", action="store_true", default=False, help="inference or not",
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
