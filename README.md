
<h2 align="center">
Effective Knowledge Graph Embedding with Quaternion Convolutional Networks
</h2>

<p align="center">
    <img src="https://img.shields.io/badge/version-1.0.1-blue">
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white">
        <a href="http://tcci.ccf.org.cn/conference/2024/"><img src="https://img.shields.io/badge/NLPCC-2024-%23bd9f65?labelColor=4aaaf1&color=4aaaf1"></a>
</p>

This repository is the official implementation of ["Effective Knowledge Graph Embedding with Quaternion Convolutional Networks"]() accepted by NLPCC 2024.

<!-- Run Locally -->
### :running: Reproduce the Results
```
-WN18RR
python run.py --data WN18RR --epoch 800 --Drop1 0.3 --Drop2 0 --Drop3 0.5 --bias --batch 256 --num_filt 300 --gpu 0 --x_ops "p.b.d" --r_ops "p.b.d" --name WN18RR --temperature 0.001

-WN18
python run.py --data WN18 --epoch 1000 --Drop1 0.3 --Drop2 0.1 --Drop3 0.2 --bias --batch 256 --num_filt 300 --gpu 0 --x_ops "p.b.d" --r_ops "p.b.d" --name WN18 --temperature 0.001

-FB15k-237
python run.py --data FB15k-237 --epoch 1000 --Drop1 0.3 --Drop2 0.2 --Drop3 0.3 --bias --batch 256 --num_filt 200 --gpu 0 --x_ops "p.b.d" --name FB15k-237 --temperature 0.007

-FB15k
python run.py --data FB15k --epoch 1000 --Drop1 0.05 --Drop2 0.5 --Drop3 0 --bias --batch 256 --num_filt 300 --gpu 0 --x_ops "p.b.d" --name FB15k --temperature 0.001 

```

## Citation

## Acknowledgement
The codes are based on [GCN4KGC](https://github.com/MIRALab-USTC/GCN4KGC) repo. Thanks for the contributions.
