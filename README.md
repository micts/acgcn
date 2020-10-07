# acgcn

Official PyTorch implementation of paper "Spot What Matters: Learning Context Using Graph Convolutional Networks for Weakly-Supervised Action Detection".

## Installation

### Requirements
`python 3.7`  
`pytorch >= 1.2.0 (CUDA 9.2, cuDNN v7.6.2)`  
`torchvision 0.4.0`    
`numpy >= 1.17`

Optionally, for visualizations    
`matplotlib`    
`scipy >= 1.3.0`       
`scikit-learn >= 0.22.2`     

### Install Requirements

Create a conda environment.

```
conda create -n myenv python=3.7
source activate myenv
```

Install requirements using

```
conda install --file requirements.txt
```

Finally, install PyTorch and torchvision.

```
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch
```

## Getting Started

Clone this repository.

``` 
git clone https://github.com/micts/acgcn.git
```

Add the repository's directory to `$PYTHONPATH` by adding the following line in `.bashrc`

```
export PYTHONPATH=/path/to/this/repo/lib:$PYTHONPATH
```

Then apply the changes.

```
source ~/.bashrc
```




