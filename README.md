#A Reinforcement Learning-based Optimization Method for Task Allocation of Agricultural Multi-Robots Clusters


## Some differences compared with the original paper

 - We use GAT as the graph embedding network.
 - We use a shared self-attention module for customer assignment.
 - The official implementation can be found [Here](https://github.com/YujiaoHu/MinMax-MTSP).
 - Paper reference:

```
Lu Z, Wang Y, Dai F, et al. A reinforcement learning-based optimization method for task allocation of agricultural multi-robots clusters[J]. Computers and Electrical Engineering, 2024, 120: 109752.

```
 - We claim that our implementation has the better performance than the original one.
## Installation
python 3.8.8

CUDA 11.1

pytorch 1.9.0 with CUDA 11.1

[PyG](https://github.com/pyg-team/pytorch_geometric) 1.7.2


[TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/)

Then install dependencies:
```
cd ~
pip install --upgrade pip
pip install torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-geometric==1.7.2
pip install ortools==9.0.9048
```

## Use code
### Training
```
python3 train.py
```
### Testing
```
python3 policy.py
```
### Generate validation and testing dataset
Adjust parameters in `data_generator.py`, then run `python3 data_generator.py`
