import numpy as np
import random
import torch
import math
import torch.nn as nn
import os
from collections import OrderedDict
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
from esrgan_1d import *
from esrgan_2d import * 
from esrgan_loss import * 
from esrgan_Dnet import *
from esrgan_Gnet import * 
from esrgan_data import *

#===variable===#
block = 23
test_data_path = "../../data/test"
cross = False

if cross == True:
    model_name = "saved_models3/generator.pth"
else:
    model_name = "saved_models2/generator.pth"

os.makedirs("test_images", exist_ok=True)

cuda = torch.cuda.is_available()
Device = 'cuda' if torch.cuda.is_available() else 'cpu'

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
dataloader = DataLoader(RealESRGANDataset(test_data_path), batch_size=1, shuffle=False, num_workers=8)

if cuda:
    net_G = Generator(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=block, num_grow_ch=32).cuda()
    criterion_MSE = torch.nn.MSELoss().cuda()

net_G.load_state_dict(torch.load(model_name))

for epoch in range(1):
    for i, img in enumerate(dataloader):
        batches_done = epoch * len(dataloader) + i
        Img_gt = Variable(img['gt'].type(Tensor)) 
        chunk_dim = 2

        a_x_split = torch.chunk(Img_gt, chunk_dim, dim=2)
        chunks_gt = []
        chunks_lq = []
        chunks_sr = []
        for cnk in a_x_split:
            cnks = torch.chunk(cnk, chunk_dim, dim=3)
            for c_ in cnks:
                chunks_gt.append(c_)

        """
            chunks_gt[]  正解画像
            chunks_lq[]  低解像度
            chunks_sr[]  生成画像
        """
        for j in range(len(chunks_gt)):

            ori_h, ori_w = chunks_gt[j].size()[2:4]
            out = chunks_gt[j]
            out = F.interpolate(out, size=(int(ori_h/4), int(ori_w/4)), mode='bicubic')
            lq = out
            chunks_lq.append(lq)
            with torch.no_grad():
                sr = net_G(lq)
            chunks_sr.append(sr)
            
        if batches_done % 1 == 0:
            up_g = torch.cat((chunks_sr[0],chunks_sr[1]),3)
            down_g = torch.cat((chunks_sr[2],chunks_sr[3]),3)
            gen_hr = torch.cat((up_g,down_g),2) 
            print(gen_hr.size())
            up_h = torch.cat((chunks_gt[0],chunks_gt[1]),3)
            down_h = torch.cat((chunks_gt[2],chunks_gt[3]),3)
            Img_hr = torch.cat((up_h,down_h),2) 
            print(Img_hr.size())
            up_l = torch.cat((chunks_lq[0],chunks_lq[1]),3)
            down_l = torch.cat((chunks_lq[2],chunks_lq[3]),3)
            Img_lr = torch.cat((up_l,down_l),2) 
            print(Img_lr.size())
        
            Img_lr = nn.functional.interpolate(Img_lr, scale_factor=4)
            PSNR1 = 10 * math.log(255*255/criterion_MSE(Img_hr, gen_hr),10)
            PSNR2 = 10 * math.log(255*255/criterion_MSE(Img_hr, Img_lr),10)
            print(PSNR1)
            print(PSNR2)
            
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            Img_lr = make_grid(Img_lr, nrow=1, normalize=True)
            Img_hr = make_grid(Img_hr, nrow=1, normalize=True)
            
            img_grid = torch.cat((Img_lr, Img_hr, gen_hr),-1)
            save_image(img_grid, "test_images/%d.png" % batches_done)

    