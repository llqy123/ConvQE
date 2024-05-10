

## Reproduce the Results
To run a model execute the following command :

```
-WN18RR
python run.py --data WN18RR --epoch 1000 --feat_drop 0 --hid_drop 0.3 --conve_hid_drop 0.5 --bias --batch 256 --num_filt 300 --gpu 0 --x_ops "p.b.d" --r_ops "p.b.d" --name WN18RR --temperature 0.001

-WN18
python run.py --data WN18 --epoch 1000 --feat_drop 0.1 --hid_drop 0.3 --conve_hid_drop 0.2 --bias --batch 256 --num_filt 300 --gpu 0 --x_ops "p.b.d" --r_ops "p.b.d" --name WN18 --temperature 0.001

-FB15k-237
python run.py --data FB15k-237 --epoch 1000 --feat_drop 0.2 --hid_drop 0.3 --conve_hid_drop 0.3 --bias --batch 256 --num_filt 200 --gpu 0 --x_ops "p.b.d" --name FB15k-237 --temperature 0.007

-FB15k
python run.py --data FB15k --epoch 1000 --feat_drop 0.5 --hid_drop 0.05 --conve_hid_drop 0 --bias --batch 256 --num_filt 300 --gpu 0 --x_ops "p.b.d" --name FB15k --temperature 0.001 

```

