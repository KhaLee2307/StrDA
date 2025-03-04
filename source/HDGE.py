import os
import itertools

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Subset

from .ops import set_grad
from .dataset import AlignCollateHDGE, hierarchical_dataset

import utils.utils_HDGE as utils

from modules.generators import define_Gen
from modules.discriminators import define_Dis


class HDGE(object):
    def __init__(self,args):

        # define the network 
        self.Gab = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, norm=args.norm, 
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
        self.Gba = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, norm=args.norm, 
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
        self.Da = define_Dis(input_nc=3, ndf=args.ndf, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)
        self.Db = define_Dis(input_nc=3, ndf=args.ndf, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)

        utils.print_networks([self.Gab,self.Gba,self.Da,self.Db], ['Gab','Gba','Da','Db'])

        # define Loss criterias
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()

        # optimizers
        self.g_optimizer = torch.optim.Adam(itertools.chain(self.Gab.parameters(),self.Gba.parameters()), lr=args.lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.Da.parameters(),self.Db.parameters()), lr=args.lr, betas=(0.5, 0.999))
        

        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)

        # try loading checkpoint
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/HDGE_gen_dis.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.Da.load_state_dict(ckpt['Da'])
            self.Db.load_state_dict(ckpt['Db'])
            self.Gab.load_state_dict(ckpt['Gab'])
            self.Gba.load_state_dict(ckpt['Gba'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 0

    def train(self,args):

        source_data, source_data_log = hierarchical_dataset(args.source_data, args, mode = "raw")
        target_data, target_data_log = hierarchical_dataset(args.target_data, args, mode = "raw")

        select_data = list(np.load(args.select_data))
        target_data_adjust = Subset(target_data, select_data)

        myAlignCollate = AlignCollateHDGE(args)

        a_loader = torch.utils.data.DataLoader(
            source_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=myAlignCollate,
            pin_memory=False,
            drop_last=True,
        )
        b_loader = torch.utils.data.DataLoader(
            target_data_adjust,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=myAlignCollate,
            pin_memory=False,
            drop_last=True,
        )

        del source_data, target_data, select_data, target_data_adjust, source_data_log, target_data_log, myAlignCollate

        a_fake_sample = utils.Sample_from_Pool()
        b_fake_sample = utils.Sample_from_Pool()

        a_loader_iter = iter(a_loader)

        for epoch in range(self.start_epoch, args.epochs):

            for b_real in tqdm(b_loader):

                try:
                    a_real = next(a_loader_iter)
                except StopIteration:
                    del a_loader_iter
                    a_loader_iter = iter(a_loader)
                    a_real = next(a_loader_iter)

                # generator Computations

                set_grad([self.Da, self.Db], False)
                self.g_optimizer.zero_grad()

                a_real = Variable(a_real)
                b_real = Variable(b_real)
                a_real, b_real = utils.cuda([a_real, b_real])

                # forward pass through generators
                a_fake = self.Gab(b_real)
                b_fake = self.Gba(a_real)

                a_recon = self.Gab(b_fake)
                b_recon = self.Gba(a_fake)

                a_idt = self.Gab(a_real)
                b_idt = self.Gba(b_real)

                # identity losses
                a_idt_loss = self.L1(a_idt, a_real) * args.lamda * args.idt_coef
                b_idt_loss = self.L1(b_idt, b_real) * args.lamda * args.idt_coef

                # adversarial losses
                a_fake_dis = self.Da(a_fake)
                b_fake_dis = self.Db(b_fake)

                real_label = utils.cuda(Variable(torch.ones(a_fake_dis.size())))

                a_gen_loss = self.MSE(a_fake_dis, real_label)
                b_gen_loss = self.MSE(b_fake_dis, real_label)

                # cycle consistency losses
                a_cycle_loss = self.L1(a_recon, a_real) * args.lamda
                b_cycle_loss = self.L1(b_recon, b_real) * args.lamda

                # total generators losses
                gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss

                # update generators
                gen_loss.backward()
                self.g_optimizer.step()

                # discriminator Computations

                set_grad([self.Da, self.Db], True)
                self.d_optimizer.zero_grad()

                # sample from history of generated images
                a_fake = Variable(torch.Tensor(a_fake_sample([a_fake.cpu().data.numpy()])[0]))
                b_fake = Variable(torch.Tensor(b_fake_sample([b_fake.cpu().data.numpy()])[0]))
                a_fake, b_fake = utils.cuda([a_fake, b_fake])

                # forward pass through discriminators
                a_real_dis = self.Da(a_real)
                a_fake_dis = self.Da(a_fake)
                b_real_dis = self.Db(b_real)
                b_fake_dis = self.Db(b_fake)
                real_label = utils.cuda(Variable(torch.ones(a_real_dis.size())))
                fake_label = utils.cuda(Variable(torch.zeros(a_fake_dis.size())))

                # discriminator losses
                a_dis_real_loss = self.MSE(a_real_dis, real_label)
                a_dis_fake_loss = self.MSE(a_fake_dis, fake_label)
                b_dis_real_loss = self.MSE(b_real_dis, real_label)
                b_dis_fake_loss = self.MSE(b_fake_dis, fake_label)

                # total discriminators losses
                a_dis_loss = (a_dis_real_loss + a_dis_fake_loss)*0.5
                b_dis_loss = (b_dis_real_loss + b_dis_fake_loss)*0.5

                # update discriminators
                a_dis_loss.backward()
                b_dis_loss.backward()
                self.d_optimizer.step()

            print("\nEpoch: (%3d/%3d) | Gen Loss: %0.4f | Dis Loss: %0.4f\n" % 
                    (epoch + 1, args.epochs, gen_loss,a_dis_loss+b_dis_loss))

            # override the latest checkpoint
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Da': self.Da.state_dict(),
                                   'Db': self.Db.state_dict(),
                                   'Gab': self.Gab.state_dict(),
                                   'Gba': self.Gba.state_dict(),
                                   'd_optimizer': self.d_optimizer.state_dict(),
                                   'g_optimizer': self.g_optimizer.state_dict()},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))

            # update learning rates
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()
