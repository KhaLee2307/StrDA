import os
from tqdm import tqdm

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from .dataset import Pseudolabel_Dataset, AlignCollateHDGE, get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DomainStratifying(object):
    def __init__(
        self, args, select_data
    ):
        """
        Stage 1: Domain Stratifying for Stratified Domain Adaptation using 2 main methods:
        - Domain Discriminator (DD)
        - Harmonic Domain Gap Estimator (HDGE)
        Each method gives one sample a distance d_i. Then, we sort them in ascending order.

        Parameters
        ----------
        args: argparse.ArgumentParser().parse_args()
            argument
        select_data: list()
            the array of selected data
        """

        self.args = args
        self.method = args.method
        self.saved_path = args.saved_path # the path to save the result of stratifying method
        self.num_subsets = args.num_subsets    # the number of subsets
        self.remain_data = select_data   # the number of remain data after selection steps
        self.k_number = len(select_data) // self.num_subsets    # the number of data point per subset
        self.distance = [] # the order of data after processing

        # make dir      
        os.makedirs(f"{self.saved_path}/{self.num_subsets}_subsets/", exist_ok=True)

    def save_subset(self):
        
        result_index = [u[0] for u in self.distance]
        result_distance = [u[1] for u in self.distance]

        if self.method == "DD":
            np.save(f"{self.saved_path}/{self.method}_{self.args.discriminator}_distance.npy", result_distance)
            np.save(f"{self.saved_path}/{self.method}_{self.args.discriminator}_index.npy", result_index)
        else:
            np.save(f"{self.saved_path}/{self.method}_{self.args.beta}_distance.npy", result_distance)
            np.save(f"{self.saved_path}/{self.method}_{self.args.beta}_index.npy", result_index)
        
        for iter in range(self.num_subsets // 2):
            # select k_number lowest distance
            add_source = [u for u in result_index[:self.k_number]]
            # select k_number highest distance
            add_target = [u for u in result_index[-self.k_number:]]
            # adjust result
            result_index = np.setdiff1d(result_index, add_source + add_target)

            # save work
            source = np.array(add_source, dtype=np.int32)
            target = np.array(add_target, dtype=np.int32)
            
            np.save(f"{self.saved_path}/{self.num_subsets}_subsets/subset_{iter + 1}.npy", source)
            np.save(f"{self.saved_path}/{self.num_subsets}_subsets/subset_{self.num_subsets - iter}.npy", target)

        if (self.num_subsets % 2 != 0):
            result_index = np.array(result_index, dtype=np.int32)
            np.save(f"{self.saved_path}/{self.num_subsets}_subsets/subset_{self.num_subsets // 2 + 1}.npy", result_index)

        print("\nAll information saved at " + f"{self.saved_path}/{self.num_subsets}_subsets/")
            
    def stratify_DD(self, adapt_data_raw, model):
        """
        Select data point for each subset and save them

        Parameters
        ----------
        adapt_data_raw: torch.utils.data.Dataset
            adapt data
        model: Model
            discriminator module for stratifying
        
        Return
        ----------
        """

        print("Start Domain Stratifying (Domain Discriminator - DD)...\n")

        unlabel_data_remain = Subset(adapt_data_raw, self.remain_data)

        # assign pseudo labels by the order of sample in dataset
        unlabel_data_remain = Pseudolabel_Dataset(unlabel_data_remain, self.remain_data)
        adapt_data_loader = get_dataloader(self.args, unlabel_data_remain, self.args.batch_size_val, shuffle=False)
        
        del unlabel_data_remain

        result = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(adapt_data_loader):
                image_tensors, index_unlabel = batch
                image = image_tensors.to(device)
                
                preds = model(image)
                preds_prob = F.sigmoid(preds).detach().cpu().squeeze().numpy().tolist()

                result.extend(list(zip(index_unlabel, preds_prob)))

            # sort result by probability in ascending order
            result = sorted(result, key=lambda x: x[1])

            print("\n20-lowest distance:")
            print(result[:20])
            
            print("\n20-highest distance:")
            print(result[-20:])

            self.distance = result

        self.save_subset()

    def stratify_HDGE(self, adapt_data_raw, dis_source, dis_target, beta):
        """
        Select data point for each subset and save them

        Parameters
        ----------
        adapt_data_raw: torch.utils.data.Dataset
            adapt data
        dis_source: Model
            discriminator of source module
        dis_target: Model
            discriminator of target module
        beta: float
            hyperparameter for HDGE method (default: 1)
        
        Return
        ----------
        """

        print("Start Domain Stratifying (Harmonic Domain Gap Estimator - HDGE)...\n")

        unlabel_data_remain = Subset(adapt_data_raw, self.remain_data)
       
        myAlignCollate = AlignCollateHDGE(self.args, infer=True)
        adapt_data_loader = torch.utils.data.DataLoader(
            unlabel_data_remain,
            batch_size=self.args.batch_size_val,
            shuffle=False,
            num_workers=self.args.workers,
            collate_fn=myAlignCollate,
            pin_memory=False,
            drop_last=False,
        )

        del unlabel_data_remain

        dis_source = dis_source.to(device)
        dis_target = dis_target.to(device)

        source_loss = []
        target_loss = []    

        dis_source.eval()
        dis_target.eval()
        with torch.no_grad():
            for batch in tqdm(adapt_data_loader):
                image_tensors = batch
                image = image_tensors.to(device)
                
                source_dis = dis_source(image)
                target_dis = dis_target(image)

                real_label = torch.ones(source_dis.size()).to(device)
                
                # calculate MSE for each sample
                source_batch_loss = torch.mean((source_dis - real_label)**2, dim=(1,2,3)).cpu().squeeze().numpy().tolist()
                target_batch_loss = torch.mean((target_dis - real_label)**2, dim=(1,2,3)).cpu().squeeze().numpy().tolist()

                source_loss.extend(source_batch_loss)
                target_loss.extend(target_batch_loss)

        np.save(f"{self.saved_path}/source_loss.npy", source_loss)
        np.save(f"{self.saved_path}/target_loss.npy", target_loss)

        # calculate di
        def formula(source_loss, target_loss, beta=1):
            return(1 + (beta)**2)*source_loss*target_loss / ((beta**2)*source_loss + target_loss)
        
        distance = [formula(s_loss, t_loss, beta) for s_loss, t_loss in zip(source_loss, target_loss)]

        result = [[index, distance] for index, distance in zip(self.remain_data, distance)]
        # sort result in ascending order
        result = sorted(result, key=lambda x: x[1])

        self.distance = result

        print("\n20-lowest distance:")
        print(result[:20])
        
        print("\n20-highest distance:")
        print(result[-20:])

        self.save_subset()
