
from dataloaders.visual_genome import VGDataLoader, VG
from lib.object_detector import ObjectDetector
import numpy as np 
from torch import optim
import torch
import pandas as pd
import time
import os
from config import ModelConfig, FG_FRACTION, RPN_FG_FRACTION, IM_SCALE, BOX_SCALE
from torch.nn import functional as F
from lib.fpn.box_utils import bbox_loss
import torch.backends.cudnn as cudnn
from lib.pytorch_misc import optimistic_restore, clip_grad_norm
from torch.optim.lr_scheduler import ReduceLROnPlateau
os.environ['CUDA_VISIBLE_DEVICES']='3'
torch.cuda.set_device(0)
cudnn.benchmark = True
conf = ModelConfig()

train, val, _ = VG.splits(num_val_im=conf.val_size, filter_non_overlap=False,
                          filter_empty_rels=False, use_proposals=False)
train_loader, val_loader = VGDataLoader.splits(train, val, batch_size=1,
                                                   num_workers=1,
                                                   num_gpus=1)
detector = ObjectDetector(classes=train.ind_to_classes, num_gpus=1,mode='refinerels',
                          use_resnet=False)
detector.cuda()

ckpt = torch.load(conf.ckpt)
optimistic_restore(detector, ckpt['state_dict'])
detector.eval()
for batch in train_loader:
    results = detector[batch]