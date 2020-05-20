"""
Miscellaneous functions that might be useful for pytorch
"""

import h5py
import numpy as np
import torch
from torch.autograd import Variable
import os
import dill as pkl
from itertools import tee
from torch import nn
from math import pi
from fasttext import load_model
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from config import W2V_MODEL, BG_VEC
import matplotlib.pyplot as plt 
import itertools
from math import floor
import logging

def optimistic_restore(network, state_dict):
    mismatch = False
    own_state = network.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print("Unexpected key {} in state_dict with size {}".format(name, param.size()))
            mismatch = True
        elif param.size() == own_state[name].size():
            own_state[name].copy_(param)
        else:
            print("Network has {} with size {}, ckpt has {}".format(name,
                                                                    own_state[name].size(),
                                                                    param.size()))
            mismatch = True

    missing = set(own_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        print("We couldn't find {}".format(',\n'.join(missing)))
        mismatch = True
    return not mismatch


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_ranking(predictions, labels, num_guesses=5):
    """
    Given a matrix of predictions and labels for the correct ones, get the number of guesses
    required to get the prediction right per example.
    :param predictions: [batch_size, range_size] predictions
    :param labels: [batch_size] array of labels
    :param num_guesses: Number of guesses to return
    :return:
    """
    assert labels.size(0) == predictions.size(0)
    assert labels.dim() == 1
    assert predictions.dim() == 2

    values, full_guesses = predictions.topk(predictions.size(1), dim=1)
    _, ranking = full_guesses.topk(full_guesses.size(1), dim=1, largest=False)
    gt_ranks = torch.gather(ranking.data, 1, labels.data[:, None]).squeeze()

    guesses = full_guesses[:, :num_guesses]
    return gt_ranks, guesses

def cache(f):
    """
    Caches a computation
    """
    def cache_wrapper(fn, *args, **kwargs):
        if os.path.exists(fn):
            with open(fn, 'rb') as file:
                data = pkl.load(file)
        else:
            print("file {} not found, so rebuilding".format(fn))
            data = f(*args, **kwargs)
            with open(fn, 'wb') as file:
                pkl.dump(data, file)
        return data
    return cache_wrapper


class Flattener(nn.Module):
    def __init__(self):
        """
        Flattens last 3 dimensions to make it only batch size, -1
        """
        super(Flattener, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)


def to_variable(f):
    """
    Decorator that pushes all the outputs to a variable
    :param f: 
    :return: 
    """
    def variable_wrapper(*args, **kwargs):
        rez = f(*args, **kwargs)
        if isinstance(rez, tuple):
            return tuple([Variable(x) for x in rez])
        return Variable(rez)
    return variable_wrapper

def arange(base_tensor, n=None):
    new_size = base_tensor.size(0) if n is None else n
    new_vec = base_tensor.new(new_size).long()
    torch.arange(0, new_size, out=new_vec)
    return new_vec


def to_onehot(vec, num_classes, fill=1000):
    """
    Creates a [size, num_classes] torch FloatTensor where
    one_hot[i, vec[i]] = fill
    
    :param vec: 1d torch tensor
    :param num_classes: int
    :param fill: value that we want + and - things to be.
    :return: 
    """
    onehot_result = vec.new(vec.size(0), num_classes).float().fill_(-fill)
    arange_inds = vec.new(vec.size(0)).long()
    torch.arange(0, vec.size(0), out=arange_inds)

    onehot_result.view(-1)[vec + num_classes*arange_inds] = fill
    return onehot_result

def save_net(fname, net):
    h5f = h5py.File(fname, mode='w')
    for k, v in list(net.state_dict().items()):
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    h5f = h5py.File(fname, mode='r')
    for k, v in list(net.state_dict().items()):
        param = torch.from_numpy(np.asarray(h5f[k]))

        if v.size() != param.size():
            print("On k={} desired size is {} but supplied {}".format(k, v.size(), param.size()))
        else:
            v.copy_(param)


def batch_index_iterator(len_l, batch_size, skip_end=True):
    """
    Provides indices that iterate over a list
    :param len_l: int representing size of thing that we will
        iterate over
    :param batch_size: size of each batch
    :param skip_end: if true, don't iterate over the last batch
    :return: A generator that returns (start, end) tuples
        as it goes through all batches
    """
    iterate_until = len_l
    if skip_end:
        iterate_until = (len_l // batch_size) * batch_size

    for b_start in range(0, iterate_until, batch_size):
        yield (b_start, min(b_start+batch_size, len_l))

def batch_map(f, a, batch_size):
    """
    Maps f over the array a in chunks of batch_size.
    :param f: function to be applied. Must take in a block of
            (batch_size, dim_a) and map it to (batch_size, something).
    :param a: Array to be applied over of shape (num_rows, dim_a).
    :param batch_size: size of each array
    :return: Array of size (num_rows, something).
    """
    rez = []
    for s, e in batch_index_iterator(a.size(0), batch_size, skip_end=False):
        print("Calling on {}".format(a[s:e].size()))
        rez.append(f(a[s:e]))

    return torch.cat(rez)


def const_row(fill, l, volatile=False):
    input_tok = Variable(torch.LongTensor([fill] * l),volatile=volatile)
    if torch.cuda.is_available():
        input_tok = input_tok.cuda()
    return input_tok


def print_para(model):
    """
    Prints parameters of a model
    :param opt:
    :return:
    """
    st = {}
    strings = []
    total_params = 0
    for p_name, p in model.named_parameters():

        if not ('bias' in p_name.split('.')[-1] or 'bn' in p_name.split('.')[-1]):
            st[p_name] = ([str(x) for x in p.size()], np.prod(p.size()), p.requires_grad)
        total_params += np.prod(p.size())
    for p_name, (size, prod, p_req_grad) in sorted(st.items(), key=lambda x: -x[1][1]):
        strings.append("{:<50s}: {:<16s}({:8d}) ({})".format(
            p_name, '[{}]'.format(','.join(size)), prod, 'grad' if p_req_grad else '    '
        ))
    return '\n {:.1f}M total parameters \n ----- \n \n{}'.format(total_params / 1000000.0, '\n'.join(strings))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def nonintersecting_2d_inds(x):
    """
    Returns np.array([(a,b) for a in range(x) for b in range(x) if a != b]) efficiently
    :param x: Size
    :return: a x*(x-1) array that is [(0,1), (0,2)... (0, x-1), (1,0), (1,2), ..., (x-1, x-2)]
    """
    rs = 1 - np.diag(np.ones(x, dtype=np.int32))
    relations = np.column_stack(np.where(rs))
    return relations


def intersect_2d(x1, x2):
    """
    Given two arrays [m1, n], [m2,n], returns a [m1, m2] array where each entry is True if those
    rows match.
    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")

    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res

def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
    v = Variable(torch.from_numpy(x).type(dtype))
    if is_cuda:
        v = v.cuda()
    return v

def gather_nd(x, index):
    """

    :param x: n dimensional tensor [x0, x1, x2, ... x{n-1}, dim]
    :param index: [num, n-1] where each row contains the indices we'll use
    :return: [num, dim]
    """
    nd = x.dim() - 1
    assert nd > 0
    assert index.dim() == 2
    assert index.size(1) == nd
    dim = x.size(-1)

    sel_inds = index[:,nd-1].clone()
    mult_factor = x.size(nd-1)
    for col in range(nd-2, -1, -1): # [n-2, n-3, ..., 1, 0]
        sel_inds += index[:,col] * mult_factor
        mult_factor *= x.size(col)

    grouped = x.view(-1, dim)[sel_inds]
    return grouped


def enumerate_by_image(im_inds):
    im_inds_np = im_inds.cpu().numpy()
    initial_ind = int(im_inds_np[0])
    s = 0
    for i, val in enumerate(im_inds_np):
        if val != initial_ind:
            yield initial_ind, s, i
            initial_ind = int(val)
            s = i
    yield initial_ind, s, len(im_inds_np)
    # num_im = im_inds[-1] + 1
    # # print("Num im is {}".format(num_im))
    # for i in range(num_im):
    #     # print("On i={}".format(i))
    #     inds_i = (im_inds == i).nonzero()
    #     if inds_i.dim() == 0:
    #         continue
    #     inds_i = inds_i.squeeze(1)
    #     s = inds_i[0]
    #     e = inds_i[-1] + 1
    #     # print("On i={} we have s={} e={}".format(i, s, e))
    #     yield i, s, e

def diagonal_inds(tensor):
    """
    Returns the indices required to go along first 2 dims of tensor in diag fashion
    :param tensor: thing
    :return: 
    """
    assert tensor.dim() >= 2
    assert tensor.size(0) == tensor.size(1)
    size = tensor.size(0)
    arange_inds = tensor.new(size).long()
    torch.arange(0, tensor.size(0), out=arange_inds)
    return (size+1)*arange_inds

def enumerate_imsize(im_sizes):
    s = 0
    for i, (h, w, scale, num_anchors) in enumerate(im_sizes):
        na = int(num_anchors)
        e = s + na
        yield i, s, e, h, w, scale, na

        s = e

def argsort_desc(scores):
    """
    Returns the indices that sort scores descending in a smart way
    :param scores: Numpy array of arbitrary size
    :return: an array of size [numel(scores), dim(scores)] where each row is the index you'd
             need to get the score.
    """
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))


def unravel_index(index, dims):
    unraveled = []
    index_cp = index.clone()
    for d in dims[::-1]:
        unraveled.append(index_cp % d)
        index_cp /= d
    return torch.cat([x[:,None] for x in unraveled[::-1]], 1)

def de_chunkize(tensor, chunks):
    s = 0
    for c in chunks:
        yield tensor[s:(s+c)]
        s = s+c

def random_choose(tensor, num):
    "randomly choose indices"
    num_choose = min(tensor.size(0), num)
    if num_choose == tensor.size(0):
        return tensor

    # Gotta do this in numpy because of https://github.com/pytorch/pytorch/issues/1868
    rand_idx = np.random.choice(tensor.size(0), size=num, replace=False)
    rand_idx = torch.LongTensor(rand_idx).cuda(tensor.get_device())
    chosen = tensor[rand_idx].contiguous()

    # rand_values = tensor.new(tensor.size(0)).float().normal_()
    # _, idx = torch.sort(rand_values)
    #
    # chosen = tensor[idx[:num]].contiguous()
    return chosen


def transpose_packed_sequence_inds(lengths):
    """
    Goes from a TxB packed sequence to a BxT or vice versa. Assumes that nothing is a variable
    :param ps: PackedSequence
    :return:
    """

    new_inds = []
    new_lens = []
    cum_add = np.cumsum([0] + lengths)
    max_len = lengths[0]
    length_pointer = len(lengths) - 1
    for i in range(max_len):
        while length_pointer > 0 and lengths[length_pointer] <= i:
            length_pointer -= 1
        new_inds.append(cum_add[:(length_pointer+1)].copy())
        cum_add[:(length_pointer+1)] += 1
        new_lens.append(length_pointer+1)
    new_inds = np.concatenate(new_inds, 0)
    return new_inds, new_lens


def right_shift_packed_sequence_inds(lengths):
    """
    :param lengths: e.g. [2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]
    :return: perm indices for the old stuff (TxB) to shift it right 1 slot so as to accomodate
             BOS toks
             
             visual example: of lengths = [4,3,1,1]
    before:
    
        a (0)  b (4)  c (7) d (8)
        a (1)  b (5)
        a (2)  b (6)
        a (3)
        
    after:
    
        bos a (0)  b (4)  c (7)
        bos a (1)
        bos a (2)
        bos              
    """
    cur_ind = 0
    inds = []
    for (l1, l2) in zip(lengths[:-1], lengths[1:]):
        for i in range(l2):
            inds.append(cur_ind + i)
        cur_ind += l1
    return inds

def clip_grad_norm(named_parameters, max_norm, clip=False, verbose=False):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    max_norm = float(max_norm)

    total_norm = 0
    param_to_norm = {}
    param_to_shape = {}
    for n, p in named_parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm ** 2
            param_to_norm[n] = param_norm
            param_to_shape[n] = p.size()

    total_norm = total_norm ** (1. / 2)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1 and clip:
        for _, p in named_parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

    if verbose:
        print('---Total norm {:.3f} clip coef {:.3f}-----------------'.format(total_norm, clip_coef))
        for name, norm in sorted(param_to_norm.items(), key=lambda x: -x[1]):
            print("{:<50s}: {:.3f}, ({})".format(name, norm, param_to_shape[name]))
        print('-------------------------------', flush=True)

    return total_norm

def update_lr(optimizer, lr=1e-4):
    print("------ Learning rate -> {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def log_margin_softmax_loss(inputs, target,margin=0.1):
    matched_inputs = inputs[torch.arange(inputs.size(0)).long().cuda(), target]
    deno = torch.sum(torch.exp(inputs), -1) - torch.exp(matched_inputs) + torch.exp(matched_inputs-margin)
    logsoftmax = torch.log(torch.exp(matched_inputs-margin)/deno)
    return -torch.mean(logsoftmax)

def get_ort_embeds(k, dims):
    ind = torch.arange(1, k+1).float().unsqueeze(1).repeat(1,dims)
    lin_space = torch.linspace(-pi, pi, dims).unsqueeze(0).repeat(k,1)
    t = ind * lin_space
    return torch.sin(t) + torch.cos(t)

def multilabel_loss(inputs, gt_rels, gt_classes):
    
    filter_inds = torch.nonzero(gt_rels[:, 3]>0).squeeze()
    filtered_gt_rels = gt_rels[filter_inds]
    ajacent_matrix = Variable(torch.eye(inputs.size(0))).cuda()
    ajacent_matrix[filtered_gt_rels[:, 1], filtered_gt_rels[:, 2]] = 1.0
    ajacent_matrix[filtered_gt_rels[:, 2], filtered_gt_rels[:, 1]] = 1.0
    multi_onehot = []
    for i in range(inputs.size(0)):
        ind = torch.nonzero(ajacent_matrix[i]).squeeze()
        multilabel = gt_classes[ind]
        vecs = Variable(torch.zeros(inputs.size(-1))).cuda()
        vecs[multilabel] = 1.0
        multi_onehot.append(vecs)
    multi_onehot = torch.stack(multi_onehot, 0)
    
    return torch.nn.functional.binary_cross_entropy_with_logits(inputs, multi_onehot)
        
def get_fast_text_embed_distances(words):
     
    fast_text_model = load_model(W2V_MODEL)
    words_np = np.array(words)
    # vecs = [np.load(BG_VEC)[:,0]]
    vecs = []
    for w in words:
        vecs.append(fast_text_model.get_word_vector(w))
    vecs = np.stack(vecs, 0)
    distances = cosine_distances(vecs)
    order = np.argsort(-distances, axis=-1)
    order_word = words_np[order]

    return distances


def plot_confusion_matrix(cm, classes,
		normalize=False,
		title='Confusion matrix',
		cmap=plt.cm.YlGn):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)
	fig = plt.figure()
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
					horizontalalignment="center",
					color="white" if cm[i, j] > thresh else "black")

	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()
	fig.canvas.draw()
	fig_arr = np.array(fig.canvas.renderer._renderer)
	plt.close()
	return fig_arr


def relation_loss(dists, labels):
    nogt_index = torch.nonzero(labels==0).cuda().squeeze()
    gt_index = torch.nonzero(labels>0).cuda().squeeze()
    norel_dists = dists[nogt_index]
    norel_labels = labels[nogt_index]
    rel_dists = dists[gt_index]
    rel_labels = labels[gt_index]

    rel_loss = torch.nn.functional.cross_entropy(norel_dists, norel_labels)
    norel_loss = torch.nn.functional.cross_entropy(rel_dists, rel_labels)

    return rel_loss, norel_loss

class LSR_Loss(nn.Module):
    def __init__(self, num_classes, only_no_gt=True, epoch=0.1):
        super(LSR_Loss, self).__init__()
        self.num_classes = float(num_classes)
        self.epoch = epoch
        self.gt_dists = torch.eye(int(self.num_classes)).float().cuda()
        ones_matrix = torch.ones_like(self.gt_dists)
        if only_no_gt:
            self.gt_dists[0] = (1-self.epoch)*self.gt_dists[0]+(self.epoch/self.num_classes)*ones_matrix[0]
        else:
            self.gt_dists = (1-self.epoch)*self.gt_dists+(self.epoch/self.num_classes)*ones_matrix

    def forward(self, dists, labels):

        num_pairs = dists.size(0)
        logsoftmax_dists = torch.nn.functional.log_softmax(dists, -1)
        gt_dists = Variable(self.gt_dists[labels.data])

        return -(gt_dists * logsoftmax_dists).sum() / num_pairs


class OHEM_Loss(nn.Module):
    def __init__(self, ratio=3, samples=256, only_no_gt=True):
        super(OHEM_Loss, self).__init__()
        self.ratio=ratio
        self.samples=samples
        self.only_no_gt=only_no_gt
        self.loss = nn.CrossEntropyLoss()

    def forward(self, dists, labels):
        if not self.only_no_gt:
            
            nogt_index = torch.nonzero(labels==0).cuda().squeeze()
            gt_index = torch.nonzero(labels>0).cuda().squeeze()
            nogt_dists = dists[nogt_index]
            nogt_nums = nogt_dists.size(0)
            nogt_labels = labels[nogt_index]
            gt_dists = dists[gt_index]
            gt_nums = gt_dists.size(0)
            gt_labels = labels[gt_index]
            selected_nums = gt_nums * self.ratio
            nogt_entropy = -F.log_softmax(nogt_dists, -1).gather(1, nogt_labels).view(-1)
            _, idx = torch.topk(nogt_entropy, nogt_nums)
            selected_idx = idx[:selected_nums]
            selected_dists = torch.cat((gt_dists, nogt_dists[selected_idx]), 0)
            selected_labels = torch.cat((gt_labels, nogt_labels[selected_idx]), 0)
            return self.loss(selected_dists, selected_labels)

        else:
            nums = dists.size(0)
            entropy = -F.log_softmax(dists, -1).gather(1, labels).view(-1)
            _, idx = torch.topk(entropy, nums)
            selected_idx = idx[:self.samples]
            return self.loss(dists[selected_idx], labels[selected_idx])

class Relation_Loss(nn.Module):
    def __init__(self, filter_items=[0, 8, 20, 21, 29, 31, 38, 44, 49]):
        super(Relation_Loss, self).__init__()
        self.filter_items = filter_items
        self.smaller_loss = torch.zeros(1).float().cuda()
        self.larger_loss = torch.zeros(1).float().cuda()

    def forward(self, dists, labels):
        larger_mask = torch.zeros_like(labels).byte()
        smaller_mask = torch.ones_like(labels).byte()
        for i in self.filter_items:
            larger_mask += labels==i 
            smaller_mask *= labels!=i 

        larger_index = torch.nonzero(larger_mask).cuda().squeeze()
        smaller_index = torch.nonzero(smaller_mask).cuda().squeeze()
        if torch.numel(smaller_index) == 0:
            smaller_loss = Variable(self.smaller_loss)
            larger_dists = dists[larger_index]
            larger_labels = labels[larger_index]
            larger_loss = torch.nn.functional.cross_entropy(larger_dists, larger_labels)
            self.larger_loss = larger_loss.data
        elif torch.numel(larger_index) == 0:
            larger_loss = Variable(self.larger_loss)
            smaller_dists = dists[smaller_index]
            smaller_labels = labels[smaller_index]
            smaller_loss = torch.nn.functional.cross_entropy(smaller_dists, smaller_labels)
            self.smaller_loss = smaller_loss.data
        else:

            larger_dists = dists[larger_index]
            larger_labels = labels[larger_index]
            smaller_dists = dists[smaller_index]
            smaller_labels = labels[smaller_index]

            larger_loss = torch.nn.functional.cross_entropy(larger_dists, larger_labels)
            smaller_loss = torch.nn.functional.cross_entropy(smaller_dists, smaller_labels)
            self.smaller_loss = smaller_loss.data
            self.larger_loss = larger_loss.data

        return larger_loss, smaller_loss

def filter_triplets(results):
    triplet_lists = results.rel_labels[:, 1:].data.cpu().numpy()
    scores = torch.nn.functional.softmax(results.rel_dists[:, 1:], -1).data.cpu().numpy().max(1)
    norel_triplets = triplet_lists[triplet_lists[:,-1]==0]
    norel_scores = scores[triplet_lists[:,-1]==0]
    norel_num = len(norel_scores)
    index = np.argsort(-norel_scores)[min(20, floor(norel_num*0.2)):]
    # scores[triplet_lists[:,-1]>0] = 1.0
    # index = np.argsort(-scores)[:120]
    filtered_triplets = np.concatenate((triplet_lists[triplet_lists[:,-1]>0], norel_triplets[index]), 0)
    return filtered_triplets

def loss_per_images(rel_dists, rel_labels):
    im_inds = rel_labels[:, 0]
    images_num = int(im_inds[-1] + 1)
    target = rel_labels[:, -1]
    indices_list = [torch.nonzero(im_inds==i).data.squeeze() for i in range(images_num)]
    losses = []
    for indices in indices_list:
        losses.append(torch.nn.functional.cross_entropy(rel_dists[indices], target[indices]))
    return sum(losses) / images_num


def order_loss(scores, labels):
    rel_labels = labels[:, -1]
    im_inds = labels[:, 0]
    num_img = int(im_inds[-1] + 1)
    res = []
    indices_set = [torch.nonzero(im_inds==i).data.squeeze() for i in range(num_img)]
    for indices in indices_set:
        cur_scores = scores[indices]
        nogt_index_np = torch.nonzero(labels[indices]==0).squeeze().numpy()
        gt_index = torch.nonzero(labels[indices]>0).cuda().squeeze()
        gt_num = gt_index.size(0)
        nogt_index = torch.from_numpy(np.random.choice(nogt_index_np,3*gt_num)).long().cuda()
        gt_scores = cur_scores[gt_index].contiguous().view(-1, 1)
        nogt_scores = cur_scores[nogt_index].contiguous().view(-1, 3)
        res.append(torch.exp(nogt_scores-gt_scores).mean())
    
    return torch.log(1 + sum(res)/len(res))


def nps_loss(inputs, gt_classes, gt_rels, im_inds, use_weight=False):

    img_nums = int(im_inds.max()) + 1
    filter_inds = torch.nonzero(gt_rels[:, 3]>0).squeeze()
    filtered_gt_rels = gt_rels[filter_inds].data.cpu().numpy()
    ajacent_matrix = np.zeros((inputs.size(0), inputs.size(0)))
    ajacent_matrix[filtered_gt_rels[:, 1], filtered_gt_rels[:, 2]] = 1.0
    ajacent_matrix[filtered_gt_rels[:, 2], filtered_gt_rels[:, 1]] = 1.0
    pairs_count = torch.from_numpy(ajacent_matrix.sum(0)).float().cuda(inputs.get_device())

    logpt = torch.nn.functional.log_softmax(inputs, dim=-1)
    logpt = logpt.gather(1, gt_classes.unsqueeze(1)).squeeze()
    pt = Variable(logpt.data.exp())

    for i in range(img_nums):
        mask = im_inds == i
        factor = 1.0 / (pairs_count[mask.data]).sum()
        pairs_count[mask.data] *= factor

    if not use_weight:
        # bounding = torch.zeros_like(pairs_count).fill_(2.0)
        # gama_powers = torch.min(, bounding)
        gama_powers = torch.clamp(-((1 - 2.0*pairs_count) ** 5.0) * torch.log(2.0*pairs_count), max=2.0)
        gama = Variable(gama_powers)
        loss = (-1 * torch.pow(1-pt, gama) * logpt).mean()
    else:
        weight = pairs_count / img_nums
        loss = (-1 * Variable(weight) * logpt).sum()



    return loss