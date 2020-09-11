# Implicit Graph Convolutional Matrix Completion (Pytorch)
Graph-based recommender system that extends [Graph Convolutional Matrix Completion](https://arxiv.org/abs/1706.02263) to the implicit feedback setting

## Note
Code base for GC-MC based from [this](https://github.com/tanimutomo/gcmc) implementation in Pytorch and Pytorch Geometric. The official implementation of original GC-MC method is [this](https://github.com/riannevdberg/gc-mc) (Tensorflow).  

## Setup
- Setup a virtual environment with python 3.6 or newer
- Install requirements (pip)
  ```
  pip install -r requirements/1.txt
  pip install --verbose --no-cache-dir -r requirements/2.txt
  pip install -r requirements/3.txt
  ```
Installation of Pytorch Geometric is very troublesome and may destroy your python environment.  
So, we strongly recommend to use a virtual environment (e.g. pyenv, virtualenv, pipenv, etc.).  
Please see [Pytorch Geometirc official document](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html) for more details.  
