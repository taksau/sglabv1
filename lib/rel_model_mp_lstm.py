"""
Let's get the relationships yo
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence
from lib.resnet import resnet_l4
from config import BATCHNORM_MOMENTUM, IM_SCALE
from lib.fpn.nms.functions.nms import apply_nms
# from lib.relationship_feat import RelationshipFeats
# from lib.decoder_rnn import DecoderRNN, lstm_factory, LockedDropout
from lib.lstm.decoder_rnn import DecoderRNN
from lib.lstm.highway_lstm_cuda.alternating_highway_lstm import AlternatingHighwayLSTM
from lib.fpn.box_utils import bbox_overlaps, center_size
from lib.get_union_boxes import UnionBoxesAndFeats
from lib.fpn.proposal_assignments.rel_assignments import rel_assignments
from lib.object_detector import ObjectDetector, gather_res, load_vgg
from lib.pytorch_misc import transpose_packed_sequence_inds, to_onehot, arange, enumerate_by_image, diagonal_inds, Flattener, get_ort_embeds, intersect_2d
from lib.sparse_targets import FrequencyBias
from lib.surgery import filter_dets
from lib.word_vectors import obj_edge_vectors
from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
from lib.self_attention_refind import Message_Passing4OBJ
import math
from lib.self_attention_refind import LayerNorm
from lib.tail_classifier import EndCell
from math import pi, atan



MODES = ('sgdet', 'sgcls', 'predcls','preddet')


def smooth_one_hot(input):
    c = (1 / pi) * atan(10) + 0.5
    diff = input[:, None, :] - input[:, :, None]
    one_hot = ((1/pi)*torch.atan(1e6*(diff + (1e-5))) + 0.5).prod(1) / c
    return one_hot

def nms_overlaps(boxes):
    """ get overlaps for each channel"""
    assert boxes.dim() == 3
    N = boxes.size(0)
    nc = boxes.size(1)
    max_xy = torch.min(boxes[:, None, :, 2:].expand(N, N, nc, 2),
                       boxes[None, :, :, 2:].expand(N, N, nc, 2))

    min_xy = torch.max(boxes[:, None, :, :2].expand(N, N, nc, 2),
                       boxes[None, :, :, :2].expand(N, N, nc, 2))

    inter = torch.clamp((max_xy - min_xy + 1.0), min=0)

    # n, n, 151
    inters = inter[:,:,:,0]*inter[:,:,:,1]
    boxes_flat = boxes.view(-1, 4)
    areas_flat = (boxes_flat[:,2]- boxes_flat[:,0]+1.0)*(
        boxes_flat[:,3]- boxes_flat[:,1]+1.0)
    areas = areas_flat.view(boxes.size(0), boxes.size(1))
    union = -inters + areas[None] + areas[:, None]
    return inters / union

def bbox_transform_inv(boxes, gt_boxes, weights=(1.0, 1.0, 1.0, 1.0)):
    """Inverse transform that computes target bounding-box regression deltas
    given proposal boxes and ground-truth boxes. The weights argument should be
    a 4-tuple of multiplicative weights that are applied to the regression
    target.

    In older versions of this code (and in py-faster-rcnn), the weights were set
    such that the regression deltas would have unit standard deviation on the
    training dataset. Presently, rather than computing these statistics exactly,
    we use a fixed set of weights (10., 10., 5., 5.) by default. These are
    approximately the weights one would get from COCO using the previous unit
    stdev heuristic.
    """
    ex_widths = boxes[:, 2] - boxes[:, 0] + 1.0
    ex_heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ex_ctr_x = boxes[:, 0] + 0.5 * ex_widths
    ex_ctr_y = boxes[:, 1] + 0.5 * ex_heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    wx, wy, ww, wh = weights
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.stack((targets_dx, targets_dy, targets_dw,
                           targets_dh), -1)
    return targets


def get_spt_features(boxes1, boxes2, boxes_u, width, height):
    # boxes_u = boxes_union(boxes1, boxes2)
    spt_feat_1 = get_box_feature(boxes1, width, height)
    spt_feat_2 = get_box_feature(boxes2, width, height)
    spt_feat_12 = get_pair_feature(boxes1, boxes2)
    spt_feat_1u = get_pair_feature(boxes1, boxes_u)
    spt_feat_u2 = get_pair_feature(boxes_u, boxes2)
    return torch.cat((spt_feat_12, spt_feat_1u, spt_feat_u2, spt_feat_1, spt_feat_2), -1)


def get_pair_feature(boxes1, boxes2):
    delta_1 = bbox_transform_inv(boxes1, boxes2)
    delta_2 = bbox_transform_inv(boxes2, boxes1)
    spt_feat = torch.cat((delta_1, delta_2[:, :2]), -1)
    return spt_feat


def get_box_feature(boxes, width, height):
    f1 = boxes[:, 0] / width
    f2 = boxes[:, 1] / height
    f3 = boxes[:, 2] / width
    f4 = boxes[:, 3] / height
    f5 = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1) / (width * height)
    return torch.stack((f1, f2, f3, f4, f5), -1)


class Boxes_Encode(nn.Module):
    def __init__(self, output_dims):
        super(Boxes_Encode, self).__init__()
        self.spt_feats = nn.Sequential(
            nn.Linear(28, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1))

    def spo_boxes(self, boxes, rel_inds):
        s_boxes = boxes[rel_inds[:, 1]]
        o_boxes = boxes[rel_inds[:, 2]]
        union_boxes = torch.cat((
            torch.min(s_boxes[:, 0:2], o_boxes[:, 0:2]),
            torch.max(s_boxes[:, 2:], o_boxes[:, 2:])
        ), 1)

        return s_boxes, o_boxes, union_boxes

    def forward(self, boxes, rel_inds):
        s_boxes, o_boxes, u_boxes = self.spo_boxes(boxes, rel_inds)
        spt_feats = get_spt_features(s_boxes, o_boxes, u_boxes, IM_SCALE, IM_SCALE)

        return self.spt_feats(spt_feats)

class LinearizedContext(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """
    def __init__(self, classes, rel_classes, mode='sgdet',
                 embed_dim=200, hidden_dim=256, obj_dim=2048,
                 nl_obj=2, nl_edge=2, dropout_rate=0.2, order='confidence',
                 pass_in_obj_feats_to_decoder=True,
                 pass_in_obj_feats_to_edge=True):
        super(LinearizedContext, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        assert mode in MODES
        self.mode = mode

        self.nl_obj = nl_obj
        self.nl_edge = nl_edge

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = obj_dim
        self.dropout_rate = dropout_rate
        self.pass_in_obj_feats_to_decoder = pass_in_obj_feats_to_decoder
        self.pass_in_obj_feats_to_edge = pass_in_obj_feats_to_edge

        assert order in ('size', 'confidence', 'random', 'leftright')
        self.order = order

        # EMBEDDINGS
        self.decoder_lin = nn.Linear(self.hidden_dim, self.num_classes)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def forward(self, obj_dists1 ,obj_feats, obj_labels=None, box_priors=None, boxes_per_cls=None):
        """
        Forward pass through the object and edge context
        :param obj_priors:
        :param obj_fmaps:
        :param im_inds:
        :param obj_labels:
        :param boxes:
        :return:
        """



            # UNSURE WHAT TO DO HERE
        if self.mode == 'predcls':
            obj_dists2 = Variable(to_onehot(obj_labels.data, self.num_classes))
        else:
            obj_dists2 = self.decoder_lin(obj_feats) + obj_dists1

        if self.mode == 'sgdet' and not self.training:
            # NMS here for baseline

            is_overlap = nms_overlaps(boxes_per_cls.data).view(
                boxes_per_cls.size(0), boxes_per_cls.size(0), boxes_per_cls.size(1)
            ).cpu().numpy() >= 0.5

            probs = F.softmax(obj_dists2, 1).data.cpu().numpy()
            probs[:, 0] = 0
            obj_preds = obj_dists2.data.new(obj_dists2.shape[0]).long().fill_(0)

            for i in range(obj_preds.size(0)):
                box_ind, cls_ind = np.unravel_index(probs.argmax(), probs.shape)
                obj_preds[int(box_ind)] = int(cls_ind)
                probs[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
                probs[box_ind] = -1.0

            obj_preds = Variable(obj_preds.view(-1))
        else:
            obj_preds = obj_labels if obj_labels is not None else obj_dists2[:,1:].max(1)[1] + 1

        return obj_dists2, obj_preds


class RelModel(nn.Module):
    """
    RELATIONSHIPS
    """
    def __init__(self, classes, rel_classes, mode='sgdet', num_gpus=1, use_vision=True, require_overlap_det=True,
                 embed_dim=200, hidden_dim=256, pooling_dim=2048,
                 nl_obj=1, nl_edge=2, use_resnet=False, order='confidence', thresh=0.01,
                 use_proposals=False, pass_in_obj_feats_to_decoder=True,
                 pass_in_obj_feats_to_edge=True, rec_dropout=0.0, use_bias=True, use_tanh=True,
                 limit_vision=True):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        :param num_gpus: how many GPUS 2 use
        :param use_vision: Whether to use vision in the final product
        :param require_overlap_det: Whether two objects must intersect
        :param embed_dim: Dimension for all embeddings
        :param hidden_dim: LSTM hidden size
        :param obj_dim:
        """
        super(RelModel, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        self.num_gpus = num_gpus
        assert mode in MODES
        self.mode = mode

        self.pooling_size = 7
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = 2048 if use_resnet else 4096
        self.pooling_dim = pooling_dim

        self.use_bias = use_bias
        self.use_vision = use_vision
        self.use_tanh = use_tanh
        self.limit_vision=limit_vision
        self.require_overlap = require_overlap_det and self.mode == 'sgdet'
        self.hook_for_grad = False
        self.gradients = []

        self.detector = ObjectDetector(
            classes=classes,
            mode=('proposals' if use_proposals else 'refinerels') if mode == 'sgdet' else 'gtbox',
            use_resnet=use_resnet,
            thresh=thresh,
            max_per_img=64,
        )
        self.ort_embedding = torch.autograd.Variable(get_ort_embeds(self.num_classes, 200).cuda())
        embed_vecs = obj_edge_vectors(self.classes, wv_dim=self.embed_dim)
        self.obj_embed = nn.Embedding(self.num_classes, self.embed_dim)
        self.obj_embed.weight.data = embed_vecs.clone()

            # This probably doesn't help it much
        self.pos_embed = nn.Sequential(*[
            nn.BatchNorm1d(4, momentum=BATCHNORM_MOMENTUM / 10.0),
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
                nn.Dropout(0.1),
        ])

        self.context = LinearizedContext(self.classes, self.rel_classes, mode=self.mode,
                                         embed_dim=self.embed_dim, hidden_dim=self.hidden_dim,
                                         obj_dim=self.obj_dim,
                                         nl_obj=nl_obj, nl_edge=nl_edge, dropout_rate=rec_dropout,
                                         order=order,
                                         pass_in_obj_feats_to_decoder=pass_in_obj_feats_to_decoder,
                                         pass_in_obj_feats_to_edge=pass_in_obj_feats_to_edge)

        # Image Feats (You'll have to disable if you want to turn off the features from here)
        self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
                                              dim=1024 if use_resnet else 512)

        self.merge_obj_feats = nn.Sequential(nn.Linear(self.obj_dim + self.embed_dim + 128, self.hidden_dim), nn.ReLU())

        # self.trans = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim//4),
        #                             LayerNorm(self.hidden_dim//4), nn.ReLU(),
        #                             nn.Linear(self.hidden_dim//4, self.hidden_dim))

        self.get_phr_feats = nn.Linear(self.pooling_dim, self.hidden_dim)

        self.embeddings4lstm = nn.Embedding(self.num_classes, self.embed_dim)

        self.lstm = nn.LSTM(input_size=self.hidden_dim+self.embed_dim, hidden_size=self.hidden_dim, num_layers=1)

        self.obj_mps1 = Message_Passing4OBJ(self.hidden_dim)
        # self.obj_mps2 = Message_Passing4OBJ(self.hidden_dim)
        self.get_boxes_encode = Boxes_Encode(64)

        if use_resnet:
            self.roi_fmap = nn.Sequential(
                resnet_l4(relu_end=False),
                nn.AvgPool2d(self.pooling_size),
                Flattener(),
            )
        else:
            roi_fmap = [
                Flattener(),
                load_vgg(use_dropout=False, use_relu=False, use_linear=pooling_dim == 4096, pretrained=False).classifier,
            ]
            if pooling_dim != 4096:
                roi_fmap.append(nn.Linear(4096, pooling_dim))
            self.roi_fmap = nn.Sequential(*roi_fmap)
            self.roi_fmap_obj = load_vgg(pretrained=False).classifier

        ###################################
        # self.obj_classify_head = nn.Linear(self.pooling_dim, self.num_classes)


        # self.post_emb_s = nn.Linear(self.pooling_dim, self.pooling_dim//2)
        # self.post_emb_s.weight = torch.nn.init.xavier_normal(self.post_emb_s.weight, gain=1.0)
        # self.post_emb_o = nn.Linear(self.pooling_dim, self.pooling_dim//2)
        # self.post_emb_o.weight = torch.nn.init.xavier_normal(self.post_emb_o.weight, gain=1.0)
        # self.merge_obj_high = nn.Linear(self.hidden_dim, self.pooling_dim//2)
        # self.merge_obj_high.weight = torch.nn.init.xavier_normal(self.merge_obj_high.weight, gain=1.0)
        # self.merge_obj_low = nn.Linear(self.pooling_dim + 5 + self.embed_dim, self.pooling_dim//2)
        # self.merge_obj_low.weight = torch.nn.init.xavier_normal(self.merge_obj_low.weight, gain=1.0)
        # self.rel_compress = nn.Linear(self.pooling_dim//2 + 64, self.num_rels, bias=True)
        # self.rel_compress.weight = torch.nn.init.xavier_normal(self.rel_compress.weight, gain=1.0)
        # self.freq_gate = nn.Linear(self.pooling_dim//2 + 64, self.num_rels, bias=True)
        # self.freq_gate.weight = torch.nn.init.xavier_normal(self.freq_gate.weight, gain=1.0)
        
        self.post_emb_s = nn.Linear(self.pooling_dim, self.pooling_dim)
        self.post_emb_s.weight = torch.nn.init.xavier_normal(self.post_emb_s.weight, gain=1.0)
        self.post_emb_o = nn.Linear(self.pooling_dim, self.pooling_dim)
        self.post_emb_o.weight = torch.nn.init.xavier_normal(self.post_emb_o.weight, gain=1.0)
        self.merge_obj_high = nn.Linear(self.hidden_dim, self.pooling_dim)
        self.merge_obj_high.weight = torch.nn.init.xavier_normal(self.merge_obj_high.weight, gain=1.0)
        self.merge_obj_low = nn.Linear(self.pooling_dim + 5 + self.embed_dim, self.pooling_dim)
        self.merge_obj_low.weight = torch.nn.init.xavier_normal(self.merge_obj_low.weight, gain=1.0)
        self.rel_compress = nn.Linear(self.pooling_dim + 64, self.num_rels, bias=True)
        self.rel_compress.weight = torch.nn.init.xavier_normal(self.rel_compress.weight, gain=1.0)
        self.freq_gate = nn.Linear(self.pooling_dim + 64, self.num_rels, bias=True)
        self.freq_gate.weight = torch.nn.init.xavier_normal(self.freq_gate.weight, gain=1.0)
        # self.ranking_module = nn.Sequential(nn.Linear(self.pooling_dim + 64, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, 1))
        if self.use_bias:
            self.freq_bias = FrequencyBias()

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    # def fixed_obj_modules(self):
    #     for p in self.detector.parameters():
    #         p.requires_grad = False
    #     for p in self.obj_embed.parameters():
    #         p.requires_grad = False
    #     for p in self.pos_embed.parameters():
    #         p.requires_grad = False
    #     for p in self.context.parameters():
    #         p.requires_grad = False
    #     for p in self.union_boxes.parameters():
    #         p.requires_grad = False
    #     for p in self.merge_obj_feats.parameters():
    #         p.requires_grad = False
    #     for p in self.get_phr_feats.parameters():
    #         p.requires_grad = False
    #     for p in self.embeddings4lstm.parameters():
    #         p.requires_grad = False
    #     for p in self.lstm.parameters():
    #         p.requires_grad = False
    #     for p in self.obj_mps1.parameters():
    #         p.requires_grad = False
    #     for p in self.roi_fmap_obj.parameters():
    #         p.requires_grad = False
    #     for p in self.roi_fmap.parameters():
    #         p.requires_grad = False


    def save_grad(self, grad):
        self.gradients.append(grad)

    def visual_rep(self, features, rois, pair_inds):
        """
        Classify the features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4]
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :param pair_inds inds to use when predicting
        :return: score_pred, a [num_rois, num_classes] array
                 box_pred, a [num_rois, num_classes, 4] array
        """
        assert pair_inds.size(1) == 2
        uboxes = self.union_boxes(features, rois, pair_inds)
        return self.roi_fmap(uboxes)

    def visual_obj(self, features, rois, pair_inds):
        assert pair_inds.size(1) == 2
        uboxes = self.union_boxes(features, rois, pair_inds)
        return uboxes

    def get_rel_inds(self, rel_labels, im_inds, box_priors):
        # Get the relationship candidates
        if self.training:
            rel_inds = rel_labels[:, :3].data.clone()
        else:
            rel_cands = im_inds.data[:, None] == im_inds.data[None]
            rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0

            # Require overlap for detection
            if self.require_overlap:
                rel_cands = rel_cands & (bbox_overlaps(box_priors.data,
                                                       box_priors.data) > 0)

                # if there are fewer then 100 things then we might as well add some?
                amt_to_add = 100 - rel_cands.long().sum()

            rel_cands = rel_cands.nonzero()
            if rel_cands.dim() == 0:
                rel_cands = im_inds.data.new(1, 2).fill_(0)

            rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)
        return rel_inds

    def union_pairs(self, im_inds):
        rel_cands = im_inds.data[:, None] == im_inds.data[None]
        rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0
        rel_inds = rel_cands.nonzero()
        rel_inds = torch.cat((im_inds[rel_inds[:,0]][:,None].data, rel_inds), -1)
        return rel_inds

    def obj_feature_map(self, features, rois):
        """
        Gets the ROI features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(
            features, rois)
        return self.roi_fmap_obj(feature_pool.view(rois.size(0), -1))

    def forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None, proposals=None, train_anchor_inds=None,
                return_fmap=False):
        """
        Forward pass for detection
        :param x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
        :param im_sizes: A numpy array of (h, w, scale) for each image.
        :param image_offset: Offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes:

        Training parameters:
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param train_anchor_inds: a [num_train, 2] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
        :return: If train:
            scores, boxdeltas, labels, boxes, boxtargets, rpnscores, rpnboxes, rellabels

            if test:
            prob dists, boxes, img inds, maxscores, classes

        """
        result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals,
                               train_anchor_inds, return_fmap=True)
        # rel_feat = self.relationship_feat.feature_map(x)

        if result.is_none():
            return ValueError("heck")

        im_inds = result.im_inds - image_offset
        boxes = result.rm_box_priors

        if self.training and result.rel_labels is None:
            assert self.mode == 'sgdet'
            result.rel_labels = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data,
                                                gt_boxes.data, gt_classes.data, gt_rels.data,
                                                image_offset, filter_non_overlap=True,
                                                num_sample_per_gt=1)

        rel_inds = self.get_rel_inds(result.rel_labels, im_inds, boxes)
        spt_feats = self.get_boxes_encode(boxes, rel_inds)
        pair_inds = self.union_pairs(im_inds)

        if self.hook_for_grad:
            rel_inds = gt_rels[:, :-1].data


        if self.hook_for_grad:
            fmap = result.fmap
            fmap.register_hook(self.save_grad)
        else:
            fmap = result.fmap.detach()

        rois = torch.cat((im_inds[:, None].float(), boxes), 1)

        result.obj_fmap = self.obj_feature_map(fmap, rois)
        # result.obj_dists_head = self.obj_classify_head(obj_fmap_rel)

        obj_embed = F.softmax(result.rm_obj_dists, dim=1) @ self.obj_embed.weight
        obj_embed_lstm = F.softmax(result.rm_obj_dists, dim=1) @ self.embeddings4lstm.weight
        pos_embed = self.pos_embed(Variable(center_size(boxes.data)))
        obj_pre_rep = torch.cat((result.obj_fmap, obj_embed, pos_embed), 1)
        obj_feats = self.merge_obj_feats(obj_pre_rep)
        # obj_feats=self.trans(obj_feats)
        obj_feats_lstm = torch.cat((obj_feats, obj_embed_lstm), -1).contiguous().view(1, obj_feats.size(0), -1)

        # obj_feats = F.relu(obj_feats)


        phr_ori = self.visual_rep(fmap, rois, pair_inds[:, 1:])
        vr_indices = torch.from_numpy(intersect_2d(rel_inds[:, 1:].cpu().numpy(), pair_inds[:, 1:].cpu().numpy()).astype(np.uint8)).cuda().max(-1)[1]
        vr = phr_ori[vr_indices]

        phr_feats_high = self.get_phr_feats(phr_ori)

        obj_feats_lstm_output, (obj_hidden_states, obj_cell_states) = self.lstm(obj_feats_lstm)

        rm_obj_dists1 = result.rm_obj_dists + self.context.decoder_lin(obj_feats_lstm_output.squeeze())
        obj_feats_output = self.obj_mps1(obj_feats_lstm_output.view(-1, obj_feats_lstm_output.size(-1)), \
                            phr_feats_high, im_inds, pair_inds)

        obj_embed_lstm1 = F.softmax(rm_obj_dists1, dim=1) @ self.embeddings4lstm.weight

        obj_feats_lstm1 = torch.cat((obj_feats_output, obj_embed_lstm1), -1).contiguous().view(1, \
                            obj_feats_output.size(0), -1)
        obj_feats_lstm_output, _ = self.lstm(obj_feats_lstm1, (obj_hidden_states, obj_cell_states))

        rm_obj_dists2 = rm_obj_dists1 + self.context.decoder_lin(obj_feats_lstm_output.squeeze())
        obj_feats_output = self.obj_mps1(obj_feats_lstm_output.view(-1, obj_feats_lstm_output.size(-1)), \
                            phr_feats_high, im_inds, pair_inds)

        # Prevent gradients from flowing back into score_fc from elsewhere
        result.rm_obj_dists, result.obj_preds = self.context(
            rm_obj_dists2,
            obj_feats_output,
            result.rm_obj_labels if self.training or self.mode == 'predcls' else None,
            boxes.data, result.boxes_all)


        obj_dtype = result.obj_fmap.data.type()
        obj_preds_embeds = torch.index_select(self.ort_embedding, 0, result.obj_preds).type(obj_dtype)
        tranfered_boxes = torch.stack((boxes[:, 0]/IM_SCALE, boxes[:, 3]/IM_SCALE, boxes[:, 2]/IM_SCALE, boxes[:, 1]/IM_SCALE, ((boxes[:, 2] - boxes[:, 0])*(boxes[:, 3]-boxes[:, 1]))/(IM_SCALE**2)), -1).type(obj_dtype)
        obj_features = torch.cat((result.obj_fmap, obj_preds_embeds, tranfered_boxes), -1)
        obj_features_merge = self.merge_obj_low(obj_features) + self.merge_obj_high(obj_feats_output)

        # Split into subject and object representations
        result.subj_rep = self.post_emb_s(obj_features_merge)[rel_inds[:, 1]]
        result.obj_rep = self.post_emb_o(obj_features_merge)[rel_inds[:, 2]]
        prod_rep = result.subj_rep * result.obj_rep



        
            # obj_pools = self.visual_obj(result.fmap.detach(), rois, rel_inds[:, 1:])
            # rel_pools = self.relationship_feat.union_rel_pooling(rel_feat, rois, rel_inds[:, 1:])
            # context_pools = torch.cat([obj_pools, rel_pools], 1)
            # merge_pool = self.merge_feat(context_pools)
            # vr = self.roi_fmap(merge_pool)
        

            # vr = self.rel_refine(vr)
            
        prod_rep = prod_rep * vr

        if self.use_tanh:
            prod_rep = F.tanh(prod_rep)

        prod_rep = torch.cat((prod_rep, spt_feats), -1)
        freq_gate = self.freq_gate(prod_rep)
        freq_gate = F.sigmoid(freq_gate)
        result.rel_dists = self.rel_compress(prod_rep)
        # result.rank_factor = self.ranking_module(prod_rep).view(-1)

        if self.use_bias:
            result.rel_dists = result.rel_dists + freq_gate * self.freq_bias.index_with_labels(torch.stack((
                result.obj_preds[rel_inds[:, 1]],
                result.obj_preds[rel_inds[:, 2]],
            ), 1))

        if self.training:
            return result

        twod_inds = arange(result.obj_preds.data) * self.num_classes + result.obj_preds.data
        result.obj_scores = F.softmax(result.rm_obj_dists, dim=1).view(-1)[twod_inds]

        # Bbox regression
        if self.mode == 'sgdet':
            bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
        else:
            # Boxes will get fixed by filter_dets function.
            bboxes = result.rm_box_priors

        rel_rep = F.softmax(result.rel_dists, dim=1)
        # rel_rep = smooth_one_hot(rel_rep)
        # rank_factor = F.sigmoid(result.rank_factor)

        return filter_dets(bboxes, result.obj_scores,
                       result.obj_preds, rel_inds[:, 1:], rel_rep)

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])

        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])

        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs