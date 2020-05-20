"""
Configuration file!
"""
import os
from argparse import ArgumentParser
import numpy as np
import logging

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data')

def path(fn):
    return os.path.join(DATA_PATH, fn)

def stanford_path(fn):
    return os.path.join(DATA_PATH, 'stanford_filtered', fn)

# =============================================================================
# Update these with where your data is stored ~~~~~~~~~~~~~~~~~~~~~~~~~

VG_IMAGES = os.path.join(DATA_PATH, 'visual_genome/VG_100K')
RCNN_CHECKPOINT_FN = path('faster_rcnn_500k.h5')
W2V_MODEL = '/home/zengjinquan/crawl-300d-2M-subword.bin'
BG_VEC = '/home/zengjinquan/no_rel.npy'

IM_DATA_FN = stanford_path('image_data.json')
VG_SGG_FN = stanford_path('VG-SGG.h5')
VG_SGG_DICT_FN = stanford_path('VG-SGG-dicts.json')
PROPOSAL_FN = stanford_path('proposals.h5')

COCO_PATH = '/home/rowan/datasets/mscoco'
# =============================================================================
# =============================================================================


MODES = ('sgdet', 'sgcls', 'predcls','preddet')

BOX_SCALE = 1024  # Scale at which we have the boxes
IM_SCALE = 592      # Our images will be resized to this res without padding

# Proposal assignments
BG_THRESH_HI = 0.5
BG_THRESH_LO = 0.0

RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
RPN_NEGATIVE_OVERLAP = 0.3

# Max number of foreground examples
RPN_FG_FRACTION = 0.5
FG_FRACTION = 0.25
# Total number of examples
RPN_BATCHSIZE = 256
ROIS_PER_IMG = 512
REL_FG_FRACTION = 0.25
RELS_PER_IMG = 256

RELS_PER_IMG_REFINE = 64

BATCHNORM_MOMENTUM = 0.01
ANCHOR_SIZE = 16

BETA = 0.9999

ANCHOR_RATIOS = (0.23232838, 0.63365731, 1.28478321, 3.15089189) #(0.5, 1, 2)
ANCHOR_SCALES = (2.22152954, 4.12315647, 7.21692515, 12.60263013, 22.7102731) #(4, 8, 16, 32)

# PREDICATES_WEIGHTS = [11602, 364, 318, 683, 933, 3030, 2689, 18159, 1518, 765, 2504, 705, 828, 1063, 30, 1507, 306, 296, 1296, 97473, 16514, 37914, 5609, 1138, 1410, 561, 172, 486, 30331, 53761, 196495, 547, 1871, 263, 1162, 624, 166, 6335, 62, 8003, 4068, 519, 6687, 822, 426, 2343, 1327, 71121, 7471, 18428]
# PREDICATES_WEIGHTS = np.array(PREDICATES_WEIGHTS, dtype=np.float64)
# PREDICATES_WEIGHTS = np.concatenate([np.array([np.sum(PREDICATES_WEIGHTS)*4]), PREDICATES_WEIGHTS], 0)
# PREDICATES_WEIGHTS = ( 1 - BETA) / (1 - np.float_power(BETA, PREDICATES_WEIGHTS))
# PREDICATES_WEIGHTS = PREDICATES_WEIGHTS.astype(np.float32)
PREDICATES_WEIGHTS = np.ones(51, dtype=np.float32)
PREDICATES_WEIGHTS[0] = 0.1


# OBJS_WEIGHTS = np.load('/home/zengjinquan/repos/neural-motifs-master/obj_sorted.npy').astype(np.float32)[:,1]
# OBJS_WEIGHTS = 1 / OBJS_WEIGHTS
# OBJS_WEIGHTS = np.concatenate([np.array([np.min(OBJS_WEIGHTS) / 100]), OBJS_WEIGHTS], 0)
# OBJS_WEIGHTS = OBJS_WEIGHTS / np.sum(OBJS_WEIGHTS)
# OBJS_WEIGHTS = 0.5 + 0.5 * OBJS_WEIGHTS

CURRICULUM = {
    0: [v for v in range(1,51)],
    1: [31 ,20 ,48 ,30 ,22 ,29 ,50 ,8 ,21 ,1],
    2: [40 ,49 ,43 ,38 ,23 ,41 ,6 ,7 ,11 ,46 ,33 ,9 ,16 ,25 ,47 ,19 ,35 ,26 ,14],
    3: [2, 3, 4, 5, 10, 12, 13, 15, 17, 18, 24, 27, 28, 32, 34, 36, 37, 39, 42, 44, 45]
}

class ModelConfig(object):
    """Wrapper class for model hyperparameters."""
    def __init__(self):
        """
        Defaults
        """
        self.coco = None
        self.ckpt = None
        self.save_dir = None
        self.lr = None
        self.batch_size = None
        self.val_size = None
        self.l2 = None
        self.clip = None
        self.num_gpus = None
        self.num_workers = None
        self.print_interval = None
        self.gt_box = None
        self.mode = None
        self.refine = None
        self.ad3 = False
        self.test = False
        self.adam = False
        self.multi_pred=False
        self.cache = None
        self.model = None
        self.use_proposals=False
        self.use_resnet=False
        self.use_tanh=False
        self.use_bias = False
        self.limit_vision=False
        self.num_epochs=None
        self.old_feats=False
        self.obj_weight=False
        self.pred_weight=False
        self.freeze_objbp=False
        self.freeze_relbp=False
        self.obj_iiloss=False
        self.rel_iiloss=False
        self.order=None
        self.det_ckpt=None
        self.nl_edge=None
        self.nl_obj=None
        self.hidden_dim=None
        self.pass_in_obj_feats_to_decoder = None
        self.pass_in_obj_feats_to_edge = None
        self.pooling_dim = None
        self.rec_dropout = None
        self.case = 0
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())



        self.__dict__.update(self.args)

        self.curriculum = CURRICULUM[self.case]

        if len(self.ckpt) != 0:
            self.ckpt = os.path.join(ROOT_PATH, self.ckpt)
        else:
            self.ckpt = None

        if len(self.cache) != 0:
            self.cache = os.path.join(ROOT_PATH, self.cache)
        else:
            self.cache = None

        if len(self.save_dir) == 0:
            self.save_dir = None
        else:
            self.save_dir = os.path.join(ROOT_PATH, self.save_dir)
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

        assert self.val_size >= 0

        if self.mode not in MODES:
            raise ValueError("Invalid mode: mode must be in {}".format(MODES))

        # if self.model not in ('motifnet', 'stanford', 'seresnet', 'motifnet-importance', \
        #                       'fusion', 'frequency', 'fusion-gpu', 'edges', 'edges-lstm', 'edges-lstm-class-head'):
        #     raise ValueError("Invalid model {}".format(self.model))


        if self.ckpt is not None and not os.path.exists(self.ckpt):
            raise ValueError("Ckpt file ({}) doesnt exist".format(self.ckpt))

    def setup_logger(self):
        name = self.save_dir.split('/')[-1]
        FORMAT = '%(levelname)s %(name)s:%(asctime)s: \n%(message)s'
        logging.root.handlers = []
        logging.basicConfig(level=logging.INFO, format=FORMAT,
                            datefmt='%m-%d %A %H:%M:%S', filename=os.path.join(self.save_dir, name + '.log'),
                            filemode='a')
        console = logging.StreamHandler()  # 定义console handler
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(FORMAT)
        console.setFormatter(formatter)
        # Create an instance
        logger = logging.getLogger(name)
        logger.addHandler(console)
        conf_params = "~~~~~~~~ Hyperparameters used: ~~~~~~~\n"
        for x, y in self.args.items():
            conf_params += "{} : {}\n".format(x, y)
        logger.info(conf_params)
        # if self.exp_details:
        #     print('Please input the experiment details and then press the Enter key:')
        #     exp_details = input()
        #     logger.info('The experiment details: {}'.format(exp_details))

        return logger

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')


        # Options to deprecate
        parser.add_argument('-coco', dest='coco', help='Use COCO (default to VG)', action='store_true')
        parser.add_argument('-ckpt', dest='ckpt', help='Filename to load from', type=str, default='checkpoints/vgdet/faster_rcnn_500k.tar')
        parser.add_argument('-det_ckpt', dest='det_ckpt', help='Filename to load detection parameters from', type=str, default='')

        parser.add_argument('-save_dir', dest='save_dir',
                            help='Directory to save things to, such as checkpoints/save', default='', type=str)

        parser.add_argument('-ngpu', dest='num_gpus', help='cuantos GPUs tienes', type=int, default=3)
        parser.add_argument('-nwork', dest='num_workers', help='num processes to use as workers', type=int, default=1)

        parser.add_argument('-lr', dest='lr', help='learning rate', type=float, default=1e-3)

        parser.add_argument('-b', dest='batch_size', help='batch size per GPU',type=int, default=2)
        parser.add_argument('-val_size', dest='val_size', help='val size to use (if 0 we wont use val)', type=int, default=5000)

        parser.add_argument('-l2', dest='l2', help='weight decay', type=float, default=1e-4)
        parser.add_argument('-clip', dest='clip', help='gradients will be clipped to have norm less than this', type=float, default=5.0)
        parser.add_argument('-p', dest='print_interval', help='print during training', type=int,
                            default=100)
        parser.add_argument('-m', dest='mode', help='mode \in {sgdet, sgcls, predcls}', type=str,
                            default='sgdet')
        parser.add_argument('-model', dest='model', help='which model to use? (motifnet, stanford). If you want to use the baseline (NoContext) model, then pass in motifnet here, and nl_obj, nl_edge=0', type=str,
                            default='motifnet')
        parser.add_argument('-old_feats', dest='old_feats', help='Use the original image features for the edges', action='store_true')
        parser.add_argument('-order', dest='order', help='Linearization order for Rois (confidence -default, size, random)',
                            type=str, default='confidence')
        parser.add_argument('-cache', dest='cache', help='where should we cache predictions', type=str,
                            default='')
        parser.add_argument('-gt_box', dest='gt_box', help='use gt boxes during training', action='store_true')
        parser.add_argument('-adam', dest='adam', help='use adam. Not recommended', action='store_true')
        parser.add_argument('-test', dest='test', help='test set', action='store_true')
        parser.add_argument('-multipred', dest='multi_pred', help='Allow multiple predicates per pair of box0, box1.', action='store_true')
        parser.add_argument('-nepoch', dest='num_epochs', help='Number of epochs to train the model for',type=int, default=25)
        parser.add_argument('-resnet', dest='use_resnet', help='use resnet instead of VGG', action='store_true')
        parser.add_argument('-proposals', dest='use_proposals', help='Use Xu et als proposals', action='store_true')
        parser.add_argument('-nl_obj', dest='nl_obj', help='Num object layers', type=int, default=1)
        parser.add_argument('-nl_edge', dest='nl_edge', help='Num edge layers', type=int, default=2)
        parser.add_argument('-hidden_dim', dest='hidden_dim', help='Num edge layers', type=int, default=256)
        parser.add_argument('-pooling_dim', dest='pooling_dim', help='Dimension of pooling', type=int, default=4096)
        parser.add_argument('-pass_in_obj_feats_to_decoder', dest='pass_in_obj_feats_to_decoder', action='store_true')
        parser.add_argument('-pass_in_obj_feats_to_edge', dest='pass_in_obj_feats_to_edge', action='store_true')
        parser.add_argument('-rec_dropout', dest='rec_dropout', help='recurrent dropout to add', type=float, default=0.1)
        parser.add_argument('-use_bias', dest='use_bias',  action='store_true')
        parser.add_argument('-use_tanh', dest='use_tanh',  action='store_true')
        parser.add_argument('-limit_vision', dest='limit_vision',  action='store_true')
        parser.add_argument('-confu_matrix', dest='confu_matrix', type=str, default='')
        parser.add_argument('-use_pred_rel', dest='use_pred_rel', action='store_true')
        parser.add_argument('-case', dest='case', type=int, default=0)
        parser.add_argument('-obj_weight', dest='obj_weight', action='store_true')
        parser.add_argument('-pred_weight', dest='pred_weight', action='store_true')
        parser.add_argument('-freeze_objbp', dest='freeze_objbp', action='store_true')
        parser.add_argument('-freeze_relbp', dest='freeze_relbp', action='store_true')
        parser.add_argument('-debug_val', dest='debug_val', action='store_true')
        parser.add_argument('-pred_recall', dest='pred_recall', type=str, default='')
        parser.add_argument('-obj_iiloss', dest='obj_iiloss', action='store_true')
        parser.add_argument('-rel_iiloss', dest='rel_iiloss', action='store_true')
        parser.add_argument('-pred_similarity_loss', dest='pred_similarity_loss', action='store_true')
        parser.add_argument('-triplets_lists', dest='triplets_lists', type=str, default='')
        parser.add_argument('-vis_topk', dest='vis_topk', type=int, default=20)
        parser.add_argument('-save_rel_recall', dest='save_rel_recall', help='dir to save relationship results', type=str, default='')
        parser.add_argument('-random_seed', dest='random_seed', type=int, default=-1)
        # parser.add_argument('-exp_details', dest='exp_details', action='store_true')
        return parser
