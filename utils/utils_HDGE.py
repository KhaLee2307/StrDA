import os
import copy

import numpy as np

import torch


# to make directories
def mkdir(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


# to make cuda tensor
def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]


def get_traindata_link(dataset_dir):
    dirs = {}
    dirs["trainA"] = os.path.join(dataset_dir, "ltrainA")
    dirs["trainB"] = os.path.join(dataset_dir, "ltrainB")
    return dirs


def get_testdata_link(dataset_dir):
    dirs = {}
    dirs["testA"] = os.path.join(dataset_dir, "ltestA")
    dirs["testB"] = os.path.join(dataset_dir, "ltestB")
    return dirs


# to save the checkpoint
def save_checkpoint(state, save_path):
    torch.save(state, save_path)


# to load the checkpoint
def load_checkpoint(ckpt_path, map_location=None):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(" [*] Loading checkpoint from %s succeed!" % ckpt_path)
    return ckpt


# to store 50 generated image in a pool and sample from it when it is full
# shrivastava et al’s strategy
class Sample_from_Pool(object):
    def __init__(self, max_elements=50):
        self.max_elements = max_elements
        self.cur_elements = 0
        self.items = []

    def __call__(self, in_items):
        return_items = []
        for in_item in in_items:
            if self.cur_elements < self.max_elements:
                self.items.append(in_item)
                self.cur_elements = self.cur_elements + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_elements)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items


class LambdaLR():
    def __init__(self, epochs, offset, decay_epoch):
        self.epochs = epochs
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epoch)/(self.epochs - self.decay_epoch)


def print_networks(nets, names):
    print("------------Number of Parameters---------------")
    i=0
    for net in nets:
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print("[Network %s] Total number of parameters : %.3f M" % (names[i], num_params / 1e6))
        i=i+1
    print("-----------------------------------------------")
