Graph Convolutional Networks in PyTorch
====

PyTorch implementation of Graph Convolutional Networks (GCNs) for semi-supervised classification [1].

For a more in-depth understanding, please see the reference section:

Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016)

This implementation makes use of the Cora dataset from [2]. This is a document graph network with about 2700 nodes and 5000 edges which includes scientific publications split into 7 categories. 

## Usage

```python train.py```

Set the parameters in the train.py file to what you like.


## Video Link

[Link to Video Presentation](https://youtu.be/bvIvccHRlYE)

## References

[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)

[2] [Sen et al., Collective Classification in Network Data, AI Magazine 2008](http://linqs.cs.umd.edu/projects/projects/lbc/)


## Repository References

[3] [Kipf & Welling TensorFlow Implementation](https://github.com/tkipf/gcn)

Code was dervied from this repository and reimplemented into this PyTorch implementation.


## Cite

```
@article{kipf2016semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1609.02907},
  year={2016}
}
```
