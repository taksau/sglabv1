export PYTHONPATH='/home/zengjinquan/for_git/neural-motifs-lab/'
echo 'PYTHONPATH=/home/zengjinquan/for_git/neural-motifs-lab/'
python models/train_rels.py -m sgcls -model motifnet -order confidence -nl_obj 2 -nl_edge 4 -b 6 -clip 5 -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/vgrel-sgcls.tar -save_dir checkpoints/motifnet-conf-sgcls-weight-fixobj -nepoch 50 -use_bias -pred_weight -freeze_objbp
