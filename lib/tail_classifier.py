import torch
from torch import nn
from torch.autograd import Variable
from lib.fpn.box_utils import bbox_overlaps, center_size
from lib.get_union_boxes import UnionBoxesAndFeats
from lib.fpn.proposal_assignments.rel_assignments import rel_assignments
from lib.pytorch_misc import transpose_packed_sequence_inds, to_onehot, arange, enumerate_by_image, diagonal_inds, Flattener, get_ort_embeds
from lib.sparse_targets import FrequencyBias
from lib.surgery import filter_dets
from lib.word_vectors import obj_edge_vectors
from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
from config import BATCHNORM_MOMENTUM, IM_SCALE
from lib.fpn.nms.functions.nms import apply_nms
import torch.nn.functional as F
from lib.object_detector import ObjectDetector, gather_res, load_vgg
import math

MODES = ('sgdet', 'sgcls', 'predcls','preddet')

class LC(nn.Module):

    def __init__(self, classes, mode='sgdet', embed_dim=20, obj_dim=4096):
        super(LC, self).__init__()
        self.classes = classes
        self.embed_dim = embed_dim
        self.obj_dim = obj_dim
        self.mode=mode
        embed_vecs = obj_edge_vectors(self.classes, wv_dim=self.embed_dim)
        self.obj_embed = nn.Embedding(self.num_classes, self.embed_dim)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.pos_embed = nn.Sequential(*[
            nn.BatchNorm1d(4, momentum=BATCHNORM_MOMENTUM / 10.0),
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        ])

        self.decoder_lin = nn.Linear(self.obj_dim + self.embed_dim + 128, self.num_classes)


    @property
    def num_classes(self):
        return len(self.classes)

    def forward(self, obj_fmaps, obj_logits, im_inds, obj_labels=None, box_priors=None, boxes_per_cls=None):
        obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed.weight
        pos_embed = self.pos_embed(Variable(center_size(box_priors)))
        obj_pre_rep = torch.cat((obj_fmaps, obj_embed, pos_embed), 1)

        if self.mode == 'predcls':
            obj_dists2 = Variable(to_onehot(obj_labels.data, self.num_classes))
        else:
            obj_dists2 = self.decoder_lin(obj_pre_rep)

        if self.mode == 'sgdet' and not self.training:
                # NMS here for baseline

            probs = F.softmax(obj_dists2, 1)
            nms_mask = obj_dists2.data.clone()
            nms_mask.zero_()
            for c_i in range(1, obj_dists2.size(1)):
                scores_ci = probs.data[:, c_i]
                boxes_ci = boxes_per_cls.data[:, c_i]

                keep = apply_nms(scores_ci, boxes_ci,
                                 pre_nms_topn=scores_ci.size(0), post_nms_topn=scores_ci.size(0),
                                 nms_thresh=0.3)
                nms_mask[:, c_i][keep] = 1

            obj_preds = Variable(nms_mask * probs.data, volatile=True)[:,1:].max(1)[1] + 1
        else:
            obj_preds = obj_labels if obj_labels is not None else obj_dists2[:,1:].max(1)[1] + 1
        
        return obj_dists2, obj_preds


class EndCell(nn.Module):

    def __init__(self, classes, num_rels, mode='sgdet', embed_dim=200, pooling_dim=4096,
                 use_bias=True):

        super(EndCell, self).__init__()
        self.classes = classes
        self.num_rels = num_rels
        assert mode in MODES
        self.embed_dim = embed_dim
        self.pooling_dim = pooling_dim
        self.use_bias = use_bias
        self.mode = mode
        self.ort_embedding = torch.autograd.Variable(get_ort_embeds(self.num_classes, self.embed_dim).cuda())
        self.context = LC(classes=self.classes, mode=self.mode, embed_dim=self.embed_dim, 
                          obj_dim=self.pooling_dim)
        self.union_boxes = UnionBoxesAndFeats(pooling_size=7, stride=16, dim=512)
        self.pooling_size=7

        roi_fmap = [
                Flattener(),
                load_vgg(use_dropout=False, use_relu=False, use_linear=pooling_dim == 4096, pretrained=False).classifier,
            ]
        if pooling_dim != 4096:
            roi_fmap.append(nn.Linear(4096, pooling_dim))
        self.roi_fmap = nn.Sequential(*roi_fmap)
        self.roi_fmap_obj = load_vgg(pretrained=False).classifier

        self.post_lstm = nn.Linear(self.pooling_dim+self.embed_dim+5, self.pooling_dim * 2)

        # Initialize to sqrt(1/2n) so that the outputs all have mean 0 and variance 1.
        # (Half contribution comes from LSTM, half from embedding.

        # In practice the pre-lstm stuff tends to have stdev 0.1 so I multiplied this by 10.
        self.post_lstm.weight.data.normal_(0, 10.0 * math.sqrt(1.0 / self.pooling_dim))
        self.post_lstm.bias.data.zero_()


        self.post_emb = nn.Linear(self.pooling_dim + self.embed_dim + 5, self.pooling_dim * 2)

        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rels, bias=True)
        self.rel_compress.weight = torch.nn.init.xavier_normal(self.rel_compress.weight, gain=1.0)
        if self.use_bias:
            self.freq_bias = FrequencyBias()

    
    @property
    def num_classes(self):
        return len(self.classes)

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

    def forward(self, last_outputs, obj_dists, rel_inds, im_inds, rois, boxes):

        twod_inds = arange(last_outputs.obj_preds.data) * self.num_classes + last_outputs.obj_preds.data
        obj_scores = F.softmax(last_outputs.rm_obj_dists, dim=1).view(-1)[twod_inds]

        rel_rep, _ = F.softmax(last_outputs.rel_dists, dim=1)[:,1:].max(1)
        rel_scores_argmaxed = rel_rep * obj_scores[rel_inds[:,0]] * obj_scores[rel_inds[:,1]]
        _, rel_scores_idx = torch.sort(rel_scores_argmaxed.view(-1), dim=0, descending=True)
        rel_scores_idx = rel_scores_idx[:100]

        filtered_rel_inds = rel_inds[rel_scores_idx.data]

        obj_fmap = self.obj_feature_map(last_outputs.fmap.detach(), rois)

        rm_obj_dists, obj_preds = self.context(
            obj_fmap,
            obj_dists.detach(),
            im_inds, last_outputs.rm_obj_labels if self.mode=='predcls' else None,
            boxes.data, last_outputs.boxes_all
        )

        obj_dtype = obj_fmap.data.type()
        obj_preds_embeds = torch.index_select(self.ort_embedding, 0, obj_preds).type(obj_dtype)
        transfered_boxes = torch.stack((boxes[:, 0]/IM_SCALE, boxes[:, 3]/IM_SCALE, boxes[:, 2]/IM_SCALE, boxes[:, 1]/IM_SCALE, ((boxes[:, 2] - boxes[:, 0])*(boxes[:, 3]-boxes[:, 1]))/(IM_SCALE**2)), -1).type(obj_dtype)
        obj_features = torch.cat((obj_fmap, obj_preds_embeds, transfered_boxes), -1)
        edge_rep = self.post_emb(obj_features)

        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.pooling_dim)

        subj_rep = edge_rep[:, 0][filtered_rel_inds[:, 1]]
        obj_rep = edge_rep[:, 1][filtered_rel_inds[:, 2]]

        prod_rep = subj_rep * obj_rep

        vr = self.visual_rep(last_outputs.fmap.detach(), rois, filtered_rel_inds[:, 1:])

        prod_rep = prod_rep * vr

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(torch.stack((
                obj_preds[filtered_rel_inds[:, 1]],
                obj_preds[filtered_rel_inds[:, 2]],
            ), 1))

        return filtered_rel_inds, rm_obj_dists, obj_preds, rel_dists