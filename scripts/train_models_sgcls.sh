#!/usr/bin/env bash

# This is a script that will train all of the models for scene graph classification and then evaluate them.


python models/train_rels.py -m sgcls -model mp-lstm -nl_obj 0 -nl_edge 0 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/vgdet/vg-24.tar \
        -save_dir checkpoints/mp-lstm -nepoch 50 -use_bias -pred_weight



