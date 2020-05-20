"""
Training script for scene graph detection. Integrated with my faster rcnn setup
"""
from dataloaders.visual_genome import VGDataLoader, VG
import numpy as np
from torch import optim
import torch
import pandas as pd
import time
import os
import random

from tensorboardX import SummaryWriter
from config import ModelConfig, BOX_SCALE, IM_SCALE, PREDICATES_WEIGHTS, CURRICULUM
from torch.nn import functional as F
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm, log_margin_softmax_loss, nps_loss
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from lib.pytorch_misc import print_para, multilabel_loss, plot_confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lib.focalloss import FocalLoss
import random

conf = ModelConfig()
focalloss = FocalLoss(gamma=2.0)
pred_weight = None 
if conf.pred_weight:
    pred_weight = torch.from_numpy(PREDICATES_WEIGHTS).float().cuda()

writer = SummaryWriter(conf.save_dir, 'loss_and_recall_record')
logger = conf.setup_logger()
if conf.model == 'mp-lstm':
    from lib.rel_model_mp_lstm import RelModel
# elif conf.model == 'edges-lstm-head':
#     from lib.rel_model_5pav1_lstm import RelModel
# elif conf.model == 'edges-lstm':
#     from lib.rel_model_5pav1_lstm_class_loss_head import RelModel
elif conf.model == 'stanford':
    from lib.rel_model_stanford import RelModelStanford as RelModel
# elif conf.model == 'fusion':
#     from lib.rel_model_fusion import RelModel
else:
    raise ValueError()

train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet')
train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)

_, test_loader = VGDataLoader.splits(train, test, mode='rel',
                                     batch_size=conf.batch_size,
                                     num_workers=conf.num_workers,
                                     num_gpus=conf.num_gpus)

detector = RelModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                    num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                    use_resnet=conf.use_resnet, order=conf.order,
                    nl_edge=conf.nl_edge, nl_obj=conf.nl_obj, hidden_dim=conf.hidden_dim,
                    use_proposals=conf.use_proposals,
                    pass_in_obj_feats_to_decoder=conf.pass_in_obj_feats_to_decoder,
                    pass_in_obj_feats_to_edge=conf.pass_in_obj_feats_to_edge,
                    pooling_dim=conf.pooling_dim,
                    rec_dropout=conf.rec_dropout,
                    use_bias=conf.use_bias,
                    use_tanh=conf.use_tanh,
                    limit_vision=conf.limit_vision
                    )

# Freeze the detector
for n, param in detector.detector.named_parameters():
    param.requires_grad = False

logger.info(print_para(detector))


def get_optim(lr):
    # Lower the learning rate on the VGG fully connected layers by 1/10th. It's a hack, but it helps
    # stabilize the models.
    fc_params = [p for n,p in detector.named_parameters() if n.startswith('roi_fmap') and p.requires_grad]
    non_fc_params = [p for n,p in detector.named_parameters() if not n.startswith('roi_fmap') and p.requires_grad]
    params = [{'params': fc_params, 'lr': lr / 10.0}, {'params': non_fc_params}]
    # params = [p for n,p in detector.named_parameters() if p.requires_grad]

    if conf.adam:
        optimizer = optim.Adam(params, weight_decay=conf.l2, lr=lr, eps=1e-3)
    else:
        optimizer = optim.SGD(params, weight_decay=conf.l2, lr=lr, momentum=0.9)

    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1,
                                  verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
    return optimizer, scheduler

# ii_rel = FocalLoss(train.num_predicates, PREDICATES_WEIGHTS, CURRICULUM, size_average=False)
# ii_obj = FocalLoss(train.num_classes, size_average=True)



ckpt = torch.load(conf.ckpt)
if conf.ckpt.split('-')[-2].split('/')[-1] == 'vgrel':
    logger.info("Loading EVERYTHING")
    start_epoch = ckpt['epoch']

    if not optimistic_restore(detector, ckpt['state_dict']):
        start_epoch = -1
        # optimistic_restore(detector.detector, torch.load('checkpoints/vgdet/vg-28.tar')['state_dict'])
else:
    start_epoch = -1
    optimistic_restore(detector.detector, ckpt['state_dict'])

    detector.roi_fmap[1][0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
    detector.roi_fmap[1][3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
    detector.roi_fmap[1][0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
    detector.roi_fmap[1][3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])

    detector.roi_fmap_obj[0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
    detector.roi_fmap_obj[3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
    detector.roi_fmap_obj[0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
    detector.roi_fmap_obj[3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])

    # detector.roi_fmap_rel[0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
    # detector.roi_fmap_rel[3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
    # detector.roi_fmap_rel[0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
    # detector.roi_fmap_rel[3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])

del ckpt 
torch.cuda.empty_cache()
detector.cuda()


def train_epoch(epoch_num):
    detector.train()
    tr = []
    start = time.time()
    for b, batch in enumerate(train_loader):
        tr.append(train_batch(batch, verbose=b % (conf.print_interval*10) == 0)) #b == 0))

        if b % conf.print_interval == 0 and b >= conf.print_interval:
            mn = pd.concat(tr[-conf.print_interval:], axis=1).mean(1)
            time_per_batch = (time.time() - start) / conf.print_interval
            logger.info("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
            print(mn)
            writer.add_scalar('train/class_loss', mn[0], b+len(train_loader)*epoch_num)
            writer.add_scalar('train/rel_loss', mn[1], b+len(train_loader)*epoch_num)
            # if 'class-head' in conf.model:
            #     # writer.add_scalar('train/recall_loss', mn[2], b + len(train_loader) * epoch_num)
            #     writer.add_scalar('train/class_loss_head', mn[2], b + len(train_loader) * epoch_num)
            #     writer.add_scalar('train/total_loss', mn[3], b + len(train_loader) * epoch_num)
            # else:
                # writer.add_scalar('train/recall_loss', mn[2], b + len(train_loader) * epoch_num)
            writer.add_scalar('train/total_loss', mn[2], b + len(train_loader) * epoch_num)
            writer.add_scalar('lr/lr0', optimizer.param_groups[0]['lr'], b+len(train_loader)*epoch_num)
            writer.add_scalar('lr/lr1', optimizer.param_groups[1]['lr'], b+len(train_loader)*epoch_num)
            writer.file_writer.flush()
            print('-----------', flush=True)
            start = time.time()
    return pd.concat(tr, axis=1)
def train_batch(b, verbose=False):
    """
    :param b: contains:
          :param imgs: the image, [batch_size, 3, IM_SIZE, IM_SIZE]
          :param all_anchors: [num_anchors, 4] the boxes of all anchors that we'll be using
          :param all_anchor_inds: [num_anchors, 2] array of the indices into the concatenated
                                  RPN feature vector that give us all_anchors,
                                  each one (img_ind, fpn_idx)
          :param im_sizes: a [batch_size, 4] numpy array of (h, w, scale, num_good_anchors) for each image.

          :param num_anchors_per_img: int, number of anchors in total over the feature pyramid per img

          Training parameters:
          :param train_anchor_inds: a [num_train, 5] array of indices for the anchors that will
                                    be used to compute the training loss (img_ind, fpn_idx)
          :param gt_boxes: [num_gt, 4] GT boxes over the batch.
          :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
    :return:
    """
    result = detector[b]

    losses = {}
    
    losses['class_loss'] = nps_loss(result.rm_obj_dists, result.rm_obj_labels, result.rel_labels, result.im_inds)
    # losses['class_loss'] = focalloss(result.rm_obj_dists, result.rm_obj_labels)
    # losses['class_loss'] = F.cross_entropy(result.rm_obj_dists, result.rm_obj_labels)
    losses['rel_loss'] = F.cross_entropy(result.rel_dists, result.rel_labels[:, -1], weight=pred_weight) * 0.25
    # losses['rank_loss'] = floss(result.rank_factor, (result.rel_labels[:, -1] > 0).float(), result.rel_labels[:, 0]) * 0.25
    # losses['recall_loss'] = recall_loss_t(result.rel_dists, result.rel_labels[:, -1], result.rel_labels[:, 0]) * 0.025
    # if 'class-head' in conf.model:
    #     losses['class_loss_head'] = focalloss(result.obj_dists_head, result.rm_obj_labels) * 0.25

    
    
    loss = sum(losses.values())

    optimizer.zero_grad()
    # if conf.freeze_objbp:
    #     # rel_losses = losses['rel_loss'] + losses['distance_loss']
    #     # rel_losses.backward()
    #     losses['rel_loss'].backward()
    # elif conf.freeze_relbp:
    #     losses['class_loss'].backward()
    loss.backward()
    clip_grad_norm(
        [(n, p) for n, p in detector.named_parameters() if p.grad is not None],
        max_norm=conf.clip, verbose=verbose, clip=True)
    losses['total'] = loss
    optimizer.step()
    res = pd.Series({x: y.data[0] for x, y in losses.items()})
    return res


def val_epoch():
    detector.eval()
    evaluator = BasicSceneGraphEvaluator.all_modes()
    for val_b, batch in enumerate(val_loader):
        val_batch(conf.num_gpus * val_b, batch, evaluator)
    evaluator[conf.mode].print_stats()
    return np.mean(evaluator[conf.mode].result_dict[conf.mode + '_recall'][100])


def val_batch(batch_num, b, evaluator):
    det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
        gt_entry = {
            'gt_classes': val.gt_classes[batch_num + i].copy(),
            'gt_relations': val.relationships[batch_num + i].copy(),
            'gt_boxes': val.gt_boxes[batch_num + i].copy(),
        }
        assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)

        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,  # hack for now.
        }

        evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )




logger.info("Training starts now!")
optimizer, scheduler = get_optim(conf.lr * conf.num_gpus * conf.batch_size)
# if conf.debug_val:
#     mAP = val_epoch()
#     print('eval pass:{}'.format(mAP))
for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):
    rez = train_epoch(epoch)
    print("overall{:2d}: ({:.3f})\n{}".format(epoch, rez.mean(1)['total'], rez.mean(1)), flush=True)
    if conf.save_dir is not None:
        torch.save({
            'epoch': epoch,
            'state_dict': detector.state_dict(), #{k:v for k,v in detector.state_dict().items() if not k.startswith('detector.')},
            # 'optimizer': optimizer.state_dict(),
        }, os.path.join(conf.save_dir, '{}-{}.tar'.format('vgrel', epoch)))

    mAp = val_epoch()
    writer.add_scalar(conf.mode+'/recall@100', mAp, epoch)
    writer.file_writer.flush()
    # mAp = 1.0
    scheduler.step(mAp)
    if any([pg['lr'] <= (conf.lr * conf.num_gpus * conf.batch_size)/9999.0 for pg in optimizer.param_groups]):
       logger.info("exiting training early")
       writer.close()
       break
writer.close()
