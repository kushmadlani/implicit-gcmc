# Implicit Graph Convolutional Matrix Completion (PyTorch)
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
Please see [Pytorch Geometric official document](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html) for more details.  

## Background 
Graph Convoluation Matrix Completion (GC-MC) views the problem of *matrix completion* on our observation matrix from the point of view of link prediction on graphs. In *GC-MC*, the interaction data is represented using bipartite user-item graph with labeled edges denoting observed ratings or interactions. Building on recent progress in deep learning on graph-structured data, the method proposes a graph auto-encoder framework based on differentiable message passing on the bipartite interaction graph. 

We make the following changes and contributions to adapt the original GC-MC method to the implicit feedback setting. These are as follows:
- Using a single edge type and processing channel to model whether that edge represents an interaction between the two nodes it connects (one user, one item). Previously each rating level was given its own processing channel.
- Weigh messages according to their confidence, as given by the weighted average of edges weights incoming to that node.
- Loss function. We change the model output from a score per rating level to a single scalar output, passed through a sigmoid nonlinearity so as to interpret it as the probability of interaction. Accordingly, our loss function became a binary cross-entropy loss vs what was previously a cross-entropy loss, where a softmax had been applied in the original presentation of the method.
- Negative interactions. We sample a number of negative unobserved user-item pairs to contribute to the loss function. Learning would not be possible without this step, since the label of all items contributing to the loss would be positive. These unobserved user-item pairs correspond to `empty' edges on the graph.


