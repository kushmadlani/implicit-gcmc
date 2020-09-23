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
Graph Convoluation Matrix Completion (GC-MC) views the problem of *matrix completion* on our observation matrix $R$ from the point of view of link prediction on graphs. In *GC-MC*, the interaction data is represented using bipartite user-item graph with labeled edges denoting observed ratings or interactions. Building on recent progress in deep learning on graph-structured data, the method proposes a graph auto-encoder framework based on differentiable message passing on the bipartite interaction graph. 

### Graph convolution encoder
The encoder of our model uses a graph convolutional layer to perform local operations that gather information from the first-order neighbourhood of a node. These are a form of message-passing, where messages that are vector-valued are passed and transformed across edges of the graph. First the message from each node is formed, where a transformation is applied to the initial node vectors, which is the same across all locations (nodes) in the graph. Each message $\mu_{a \to b} \in \mathbb{R}^h$ has dimension $h$, a hyperparameter of the model. Messages take the following form, here from item node $i$ to user node $u$:

$$
\begin{aligned}
\mu_{i \to u} = Wo_i 
\end{aligned}
$$

where $W \in \mathbb{R}^{h \times d}$ is a parameter matrix that is to be learned during training, which maps the initial feature vector $o_i \in \mathbb{R}^d$ into $\mu_{i \to u} \in \mathbb{R}^h$. Messages from users to items $\mu_{u \to i}$ are calculated in an analogous way.

After the message-passing step, we sum the incoming messages to each node, multiplying each message by weight $\bar{c}_{ui}$ to result in a single intermediate vector denoted $h_i$, that represents a weighted average of messages. This is referred to as the *graph convolutional layer*:

$$
\begin{aligned}
h_u = \sigma \big( \sum_{i \in \mathcal{N}_u} \bar{c}_{ui} \mu_{i \to u} \big) 
\end{aligned}
$$

Here $\mathcal{N}_u$ denotes the neighbours of node $n_u$ and $\sigma(\cdot)$ denotes an element-wise non-linear activation function chosen to be $ReLU(\cdot)=\max(0,\cdot)$. We normalise each message with respect to the in-degree of each node where each message is weighed by the edge label (i.e. the confidence parameter $c_{ui}$ associated with the observation between user $u$ and $i$) :

$$
\begin{aligned}
\bar{c}_{ui} = \frac{c_{ui}}{\sum_{i \in \mathcal{N}_u} c_{ui}} 
\end{aligned}
$$

The motivation for use of such a weighting is so that the messaging passing step forms a *weighted average* of all incoming messages to that node. To arrive at the final embedding of user node $u$ we passs the intermediate output $h_u$ through a fully-connected *dense layer*:

$$
\begin{aligned}
    x_u = \sigma(W'h_u) 
\end{aligned}
$$

where $W' \in \mathbb{R}^{f \times h}$ is a separate parameter matrix to be learnt during training. The item embedding $y_i$ is calculated analogously with the same parameter matrix $W'$ or if side-information is included we use separate parameter matrices for user and item nodes.

We mention here, as in the original *GC-MC* paper that instead of a simple linear message transformation, variations are possible such as $\mu_{i \to u}=nn(o_u,o_i)$ where $nn$ is a neural network itself. In addition, instead of choosing a specific normalisation constant for individual messages, one could deploy some a form of attention mechanism, explored in some [recent work](https://www.sciencedirect.com/science/article/abs/pii/S0950705120304196).

### Bilinear decoder
To reconstruct links in the bipartite interaction graph the decoder produces a probability of user $u$ and item $i$ interacting as follows:

$$
\begin{aligned}
\hat{a}_{ui} = p(A_{ui} > 0) = \sigma(x_u^T Q y_i)
\end{aligned}
$$

where $Q$ is a trainable parameter matrix of shape $f \times f$. Here $\sigma(x)=1/(1+e^{-x})$ is the usual sigmoid function that maps the bilinear form into $[0,1]$ so that we gain a probabilistic interpretation of our output.

### Training *iGC-MC*

We train this graph auto-encoder by minimising the log likelihood of the predicted ratings $\hat{A}_{ij}$. Unlike with explicit feedback, our loss is not only calculated over observed user-item pairs in $\mathcal{O}$. In the implicit feedback setting we need to account for the binary nature of interactions by sampling a number of negative user-item pairs from $\mathcal{O}^-$. The number of negative samples per positive sample is an integer hyperparameter $c$. We call this combined set of positive and negative samples $\mathcal{S}$. Thus our objective function, equivalent to a binary cross-entropy loss, which we seek to minimise is:

$$
\begin{aligned}
\mathcal{L} &= -\sum_{u,i \in \mathcal{S}} p_{ui} \log{\hat{a}_{ui}} + (1-p_{ui})\log{(1-\hat{a}_{ui})}
\end{aligned}
$$

where $p_{ui} \in \{0,1\}$ is the true interaction between user $u$ and item $i$, whilst $\hat{a}_{ui}$ is our model's output - a probability of interaction.

