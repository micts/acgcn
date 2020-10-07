# acgcn

Official PyTorch implementation of paper "Spot What Matters: Learning Context Using Graph Convolutional Networks for Weakly-Supervised Action Detection".

## Requirements    
`python 3.7`  
`pytorch >= 1.2.0 (CUDA 9.2, cuDNN v7.6.2)`  
`torchvision 0.4.0`    
`numpy >= 1.17`

Optionally, for visualizations    
`matplotlib`    
`scipy >= 1.3.0`       
`scikit-learn >= 0.22.2`     

## Installation

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

Add the repository's directory to `$PYTHONPATH` by adding the following line to `.bashrc`

```
export PYTHONPATH=/path/to/this/repo/lib:$PYTHONPATH
```

Then apply the changes.

```
source ~/.bashrc
```

## Data Preperation

We use the Daily Action on Localization in Youtube (DALY) dataset: http://thoth.inrialpes.fr/daly/index.php.

### Download

See instructions on how to download DALY: http://thoth.inrialpes.fr/daly/getdaly.php. 

**Important Note**    
Several videos are not available on Youtube anymore, and others have different resolution than the original videos. It is recommended to download the videos from Inria's cache, which contains the original videos. You can access the cache at http://thoth.inrialpes.fr/daly/requestaccess.php. It is recommended to check that all videos are available in the cache by matching them with those contained in the annotations (`/data/DALY/annotations/daly1.1.0.pkl`). In case there are missing videos, you can always try to download them from Youtube, see link for instructions above.





