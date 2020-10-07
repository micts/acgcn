# acgcn

Official PyTorch implementation of paper "Spot What Matters: Learning Context Using Graph Convolutional Networks for Weakly-Supervised Action Detection".

acgcn -- Actor-Context/Centric Graph Convolutional Networks. 

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

Create a conda environment:

```
conda create -n myenv python=3.7
source activate myenv
```

Install requirements using:

```
conda install --file requirements.txt
```

Finally, install PyTorch and torchvision:

```
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch
```



## Getting Started

Clone this repository:

``` 
git clone https://github.com/micts/acgcn.git
```

Add the repository's directory to `$PYTHONPATH` by adding the following line to your `.bashrc` file:

```
export PYTHONPATH=/path/to/this/repo/lib:$PYTHONPATH
```

Finally, apply the changes:

```
source ~/.bashrc
```

## Data Preperation

We use the Daily Action on Localization in Youtube (DALY) dataset: http://thoth.inrialpes.fr/daly/index.php.

### Download Videos

See [instructions](http://thoth.inrialpes.fr/daly/getdaly.php) on how to download DALY.

**Important Note**    
Several videos of DALY are not available on Youtube anymore, and others have different resolution than the original videos. It is recommended to download the original videos from Inria's cache, which can be accessed at http://thoth.inrialpes.fr/daly/requestaccess.php. It is recommended to check that all videos are available in the cache by matching them with those contained in the annotations (`/data/DALY/annotations/daly1.1.0.pkl`). In case there are missing videos, you can always try to download them from Youtube, see link for instructions above.

### Extract Frames

Assuming that the downloaded videos are placed in `data/DALY/DALY_videos`, we extract the frames from each video using:

```
cd utils/
./extract_frames.sh ../data/DALY/DALY_videos ../data/DALY/DALY_frames 
```

The above command uses `ffmpeg` to extract the frames, and saves them under `data/DALY/DALY_frames`.

### Resize Frames

We resize all frames to 224x224 pixels:

```
python resize_frames.py ../data/DALY/DALY_frames ../data/DALY/frames 224 224
```

`data/DALY/frames` now contains the resized frames. We can remove the original frames and videos:

``` 
cd ..
rm -rf data/DALY/DALY_frames data/DALY/DALY_videos
```

### File Structure













