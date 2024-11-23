# FairNS: Fair Negative Sampling in Implicit Collaborative Filtering via Diffusion-based Models
We provide the code (in pytorch) and datasets for our paper: "Fair Negative Sampling in Implicit Collaborative Filtering via Diffusion-based Models" (FairNS)


#### Environment Requirement

The code has been tested running under Python 3.8.19. The required packages are as follows:

- pytorch==2.4.0
- torch-geometric==2.5.3
- numpy==1.24.3
- scipy == 1.10.1
- scikit-learn==1.3.2
- prettytable==3.11.0



#### Training

The training commands are as following:

```
python main.py --dataset ml-2-types --gnn lightgcn --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 2 --context_hops 0 --ns rns --alpha 0.1 --beta 0.4 --n_negs 1 --K 2 --run_type 4 --rand_type 1 --d_weight 0.01
```

```
python main.py --dataset ml-4-types --gnn lightgcn --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 2 --context_hops 0 --ns rns --alpha 0.1 --beta 0.4 --n_negs 1 --K 2 --run_type 4 --rand_type 1 --d_weight 0.01
```

```
python main.py --dataset tenrec-2 --gnn lightgcn --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 2 --context_hops 0 --ns rns --alpha 0.1 --beta 0.4 --n_negs 1 --K 2 --run_type 4 --rand_type 1 --d_weight 0.01
```

```
python main.py --dataset tenrec-4 --gnn lightgcn --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 2 --context_hops 0 --ns rns --alpha 0.1 --beta 0.4 --n_negs 1 --K 2 --run_type 4 --rand_type 1 --d_weight 0.01
```

#### Datasets

We use four processed datasets: MovieLens-2-types, MovieLens-4-types, Tenrec-2 and Tenrec-4.



|          | \#user | \#item | \#inter. |
|:--------:|:------:|:------:|:--------:|
|   ML-2   |  3798  |  503   |  45526   |
|   ML-4   | 560700 |  1048  |  124829  |
| Tenrec-2 |  8170  |  1180  |  57272   |
| Tenrec-4 |  9002  |  1624  |  79926   |


