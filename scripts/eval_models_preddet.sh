
export PYTHONPATH='/home/zengjinquan/repos/neural-motifs-master'

python python models/eval_rels.py -m preddet -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet-sgcls/vgrel-7.tar -nepoch 50 -use_bias -cache motifnet_preddet