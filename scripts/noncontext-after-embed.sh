export PYTHONPATH='/home/zengjinquan/for_git/neural-motifs-lab/'
echo 'PYTHONPATH=/home/zengjinquan/for_git/neural-motifs-lab/'
python models/train_rels.py -m sgcls -model motifnet -order confidence -nl_obj 0 -nl_edge 0 -b 6 -clip 5 -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/motifnet-conf-sgcls-noncontext-with-pred-embed/vgrel-15.tar -save_dir checkpoints/motifnet-conf-sgcls-noncontext-after-embed -nepoch 50 -use_bias
