import random
import numpy as np
from argparse import ArgumentParser

import torch
import torch.backends.cudnn as cudnn

import utils.utils_HDGE as utils

from modules.discriminators import define_Dis

import source.HDGE as md
from source.stratify import DomainStratifying
from source.dataset import hierarchical_dataset


# to get arguments from commandline
def get_args():
    """ Argument """
    parser = ArgumentParser()
    parser.add_argument(
        "--source_data",
        default="data/train/synth/",
        help="path to source dataset",
    )
    parser.add_argument(
        "--target_data",
        default="data/train/real/",
        help="path to target dataset",
    )
    parser.add_argument(
        "--select_data",
        required=True,
        help="path to select data",
    )
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--decay_epoch', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--batch_size_val', type=int, default=128)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--load_height', type=int, default=48)
    parser.add_argument('--load_width', type=int, default=160)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--crop_height', type=int, default=32)
    parser.add_argument('--crop_width', type=int, default=100)
    parser.add_argument('--lamda', type=int, default=10)
    parser.add_argument('--idt_coef', type=float, default=0.5)
    parser.add_argument('--training', action='store_true', default=False)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--checkpoint_dir', type=str, default='./HDGE/checkpoints/')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    """ Adaptation """
    parser.add_argument("--num_subsets", type=int, required = True, help="hyper-parameter n, number of subsets partitioned from target domain data")
    parser.add_argument("--method", required=True, help="select Domain Stratifying method, DD|HDGE")
    parser.add_argument("--beta", type=float, required = True, help="hyper-parameter beta in HDGE formula")
    parser.add_argument("--infer", action='store_true', default=False, help='inference or not')
    parser.add_argument("--infer_source", action='store_true', default=False, help='inference source')
    """ Experiment """ 
    parser.add_argument(
        "--manual_seed", type=int, default=111, help="for random seed setting"
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    """ Seed and GPU setting """
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)

    cudnn.benchmark = True  # it fasten training
    cudnn.deterministic = True

    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    print(not args.no_dropout)
    if args.training:
        print("Training")
        model = md.HDGE(args)
        model.train(args)
    if args.infer:
        print("Inference")

        # load target data (raw)
        adapt_data_raw, _ = hierarchical_dataset(args.target_data, args, mode = "raw")

        if args.infer_source == False:
            select_data = list(np.load(args.select_data))
        else:
            select_data = list(range(len(adapt_data_raw)))

        # setup Harmonic Domain Gap Estimator (HDGE)
        hdge = DomainStratifying(args, select_data)

        dis_source = define_Dis(input_nc=3, ndf=args.ndf, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)
        dis_target = define_Dis(input_nc=3, ndf=args.ndf, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)

        utils.print_networks([dis_source,dis_target], ['Da','Db'])

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            dis_source.load_state_dict(ckpt['Da'])
            dis_target.load_state_dict(ckpt['Db'])
        except:
            print(' [*] No checkpoint!')

        # Domain Stratifying
        hdge.stratify_HDGE(adapt_data_raw, dis_source, dis_target, args.beta)


if __name__ == '__main__':
    main()
