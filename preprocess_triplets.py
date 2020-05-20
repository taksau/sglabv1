import numpy as np
import dill as pkl
from argparse import ArgumentParser

parser = ArgumentParser(description='the path of source and destination.')
parser.add_argument('-source_file', dest='source', type=str, default=None)
parser.add_argument('-destin_file', dest='destin', type=str, default=None)
parser.add_argument('-freq_array', dest='freq', type=str, default=None)
args = parser.parse_args()

assert args.source is not None
assert args.destin is not None

def not_intersect_2d(x1, x2):
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
    res = (x1[..., None] != x2.T[None, ...]).all(1)
    return res


source_file = open(args.source, 'rb')
source_data = pkl.load(source_file)
source_file.close()
destin_data = {}
top3_label = np.array([31, 20, 48], dtype=np.int64)
selected_label_freq = np.zeros((51,), dtype=np.int64)
for key in source_data.keys():
    pred_triplets = source_data[key]['pred_triplets']
    scores = source_data[key]['triplets_scores']
    gt_relations = source_data[key]['gt_relations']
    prc = not_intersect_2d(pred_triplets[:,:2], gt_relations[:, :2])
    index = np.min(prc, 1)
    pred_triplets = pred_triplets[index]
    scores = scores[index]
    prc = not_intersect_2d(np.expand_dims(pred_triplets[:,2],1), np.expand_dims(top3_label, 1))
    index = np.min(prc, 1)
    pred_triplets = pred_triplets[index]
    scores = scores[index]
    if gt_relations.shape[0] > 3:
        selected_label = np.zeros((0,3), dtype=gt_relations.dtype)
    else:
        selected_label = pred_triplets[:1]
        if np.size(selected_label) > 0:
            selected_label_freq[selected_label[0][2]] += 1
    destin_data[key]={
        'pred_triplets': pred_triplets,
        'triplets_scores': scores,
        'selected_label': selected_label,
    }

with open(args.destin, 'wb') as f:
    pkl.dump(destin_data, f)
np.save(args.freq, selected_label_freq)
