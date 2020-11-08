# acgcn

Official PyTorch implementation of paper "Spot What Matters: Learning Context Using Graph Convolutional Networks for Weakly-Supervised Action Detection". Submitted to *International Workshop on Deep Learning for Human Activity Understanding (DL-HAU2020)* - ICPR 2020.

acgcn -- Actor-Context/Centric Graph Convolutional Networks. 

<p align="middle">
  <img src="images/framework_overview.png" width="80%" >
  <img src="images/drinking.png" width="40%" />
  <img src="images/brushing_teeth.png" width="40%" /> 
  <img src="images/cleaning_floor.png" width="40%" />
  <img src="images/cleaning_windows.png" width="40%" />
  <img src="images/ironing.png" width="40%" />
  <img src="images/taking_photos.png" width="40%" />
</p>


## Table of Contents  
* [Requirements](#requirements)  
* [Getting Started](#gettingstarted)
* [Installation](#installation)
* [Data Preparation](#datapreparation)
  * [Download Videos](#downloadvideos)
  * [Extract Frames](#extractframes)
  * [Resize Frames](#resizeframes)
  * [File Structure](#filestructure)
* [How to Use](#howtouse)
  * [Training](#training)
  * [Inference](#inference)
  * [Results](#results)  
* [Acknowledgments](#acknowledgments)


## Requirements<a name="requirements"/>          

`python 3.7`  
`pytorch 1.2.0 (CUDA 9.2, cuDNN v7.6.2)`  
`torchvision 0.4.0`    
`numpy >= 1.17`  
`pillow`

Optionally, for visualizations:  
`matplotlib`    
`scipy >= 1.3.0`       
`scikit-learn >= 0.22.2`     


## Getting Started<a name="gettingstarted"/>      

Create a conda environment:

```
conda create -n myenv python=3.7
source activate myenv
```

For older versions of `conda`, you might need to use `conda` instead of `source`:

```
conda activate myenv
```

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

## Installation<a name="installation"/>      

Install requirements:

```
conda install --file requirements.txt
```

Finally, install PyTorch and torchvision:

```
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch
```

## Data Preperation<a name="datapreparation"/>          

We use the Daily Action on Localization in Youtube (DALY) dataset: http://thoth.inrialpes.fr/daly/index.php.


### Download Videos<a name="downloadvideos"/>     

See [instructions](http://thoth.inrialpes.fr/daly/getdaly.php) on how to download DALY.

**Important Note**    
Several videos of DALY are not available on Youtube anymore, and others have different resolution than the original videos. It is recommended to download the original videos from Inria's cache, which can be accessed at http://thoth.inrialpes.fr/daly/requestaccess.php. It is recommended to check that all videos are available in the cache by matching them with those contained in the annotations (`/data/DALY/annotations/daly1.1.0.pkl`). In case there are missing videos, you can always try to download them from Youtube, see link for instructions above.


### Extract Frames<a name="extractframes"/>        

Assuming that the downloaded videos are placed in `data/DALY/DALY_videos`, we extract the frames from each video using:

```
cd utils/
./extract_frames.sh ../data/DALY/DALY_videos ../data/DALY/DALY_frames 
```

The above command uses `ffmpeg` to extract the frames, and saves them under `data/DALY/DALY_frames`.


### Resize Frames<a name="resizeframes"/>    

We resize all frames to 224x224 pixels:

```
python resize_frames.py ../data/DALY/DALY_frames ../data/DALY/frames 224 224
```

`data/DALY/frames` now contains the resized frames. We can remove the original frames and videos:

``` 
cd ..
rm -rf data/DALY/DALY_frames data/DALY/DALY_videos
```

### Download Human Tubes and Annotations

Download all required human tubes and annotations as a .pkl file (`annotated_data.pkl`) using the following [link](https://drive.google.com/uc?export=download&id=1taQvy56vEuCAT2-4UeBFfhxsZEw7MXlG). Place `annotated_data.pkl` in `data/DALY/annotations`. 

### File Structure<a name="filestructure"/>    

The file structure under `data/DALY/` should be as follows:

```
DALY/
|_ annotations
|  |_ all_training_videos.pkl
|  |_ all_videos.pkl
|  |_ annotated_data.pkl
|  |_ daly1.1.0.pkl
|  |_ frames_size.pkl
|  |_ training_videos.pkl
|  |_ validation_videos.pkl
|_ frames  
|  |_ video_name_0.mp4
|  |  |_ frame000001.jpg
|  |  |_ frame000002.jpg
|  |  |_ ...
|  |_ video_name_1.mp4
|  |  |_ frame000001.jpg
|  |  |_ frame000002.jpg
|  |  |_ ...
```


## How to Use<a name="howtouse"/>    

In the following, we describe the process of training and testing the Baseline model and the GCN model. Both models use I3D as a backbone (pre-trained on Imagenet + Kinetics-400) and augmented with a RoI pooling layer for action detection. The GCN model further incorporates Graph Convolutional Networks to model actor-context relations as actor-object and actor-actor interactions.   

### Training<a name="training"/>    

Train a GCN model:
```
sh run/train_gcn.sh
```

Train a Baseline model:
```
sh run/train_baseline.sh
```

Both commands above run `main.py`. For help and a description of input arguments:

```
python tools/train.py -h
```

Every run is assigned a unique identifier/filename of the form `yyyy-mm-dd_hh-mm-ss`, i.e. `2020-03-25_23-56-26`. Results, such as model's training history, are saved under `results/model_name/filename`, for example `results/gcn/2020-03-25_23-56-26`. The directory contains a pickle file with the content of `config.py` (model's hyperparameters and other configuration settings), and a series of `.pth` files. A `.pth` file is of the form `epoch_x_loss.pth`, where `x` indicates the epoch number, and `loss` corresponds to the epoch's validation loss. e.g. `epoch_300_1.108.pth`. A `.pth` file contains, for a given epoch, saved model's weights, epoch's training and validation loss, and evaluation metric results. The files are saved every *x*-number of epochs (see `num_epochs_to_val` in `config.py`). 


### Inference<a name="inference"/>      

Make sure to replace the values inside the following scripts with the model's information (model id, path to model weights, path to config file, etc.) to be used for inference. Use either a model you trained (model weights and config files are located in `results/`), or use one of the trained models already provided (see [Results](#results)).



to be used for inference.  

Inference on the test set using a GCN model:

```
sh run/test_gcn.sh
```

Inference using a Baseline model:

```
sh run/test_baseline.sh
```

Both commands above run `inference.py`. For help and a description of input arguments:

```
python tools/inference.py -h
```

### Results<a name="results"/>     

We provide trained models (model weights and config files) for GCN and Baseline. There are 5 models for GCN and Baseline respectively, using the same hyperparameters, but with different initialized seeds. Models and results correspond to those reported in the paper. Average Test Video-mAP (<span>&#177;</span> std) across the five repetitions is 61.82 (<span>&#177;</span> 0.51) for GCN and 59.58 (<span>&#177;</span> 0.22) for Baseline. 

#### GCN

| Model ID | Num Layers | Num Graphs | Merge Function | Test Video-mAP | Model and Config |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| 2020-03-22_12-50-27 | 2 | 2 | Concat | 62.73 | [Weights](https://drive.google.com/uc?export=download&id=1zqnuEfwsPDV-04K7XRL5zi-JH9fKFO1w), [Config](https://drive.google.com/uc?export=download&id=1cg0ITYgpOiN7EImDbRHRqmYt3xBw7zjJ) |
| 2020-03-24_15-15-49 | 2 | 2 | Concat | 61.14 | [Weights](https://drive.google.com/uc?export=download&id=1VMcM3Ve1-W_PnJTHFk92WLFtb1ZrnAwJ), [Config](https://drive.google.com/uc?export=download&id=1fC2AXMlBC5T1-rfFVhsI3Il9ssaa3DWQ) |
| 2020-03-24_15-13-25 | 2 | 2 | Concat | 61.82 | [Weights](https://drive.google.com/uc?export=download&id=1chCEzdNHeQwZjqSFZX9r65uvxnB1q2fe), [Config](https://drive.google.com/uc?export=download&id=1Ir70MIIrKIigHuHvxvE_egcXdguOWZky) |
| 2020-03-25_23-56-26 | 2 | 2 | Concat | 61.8 | [Weights](https://drive.google.com/uc?export=download&id=1Id-PmuK_3D5rBB5rlQREjbLWqVXpId89), [Config](https://drive.google.com/uc?export=download&id=1NrgHqFfmMlZ0E-lukIZTRaHm2n6TgkTJ) |
| 2020-03-26_16-04-26 | 2 | 2 | Concat | 61.64 | [Weights](https://drive.google.com/uc?export=download&id=1vdrqNWG5F53vRPwXbZkcf57TSLLYoPfu), [Config](https://drive.google.com/uc?export=download&id=1dlJH2Ji1Itan0_JeVkyRn8UZ0osI6TSw) |

#### Baseline

 Model ID | Test Video-mAP | Model and Config |
| :-------------: | :-------------: | :-------------: |
| 2020-03-07_21-32-26 | 59.54 | [Weights](https://drive.google.com/uc?export=download&id=1XKPnvH3BgOiRH7OV2fIBJGaEsq9J5NKo), [Config](https://drive.google.com/uc?export=download&id=1mTRdwQ1zUiN4_iOfNtxw8w8jHDjOv2-H) |
| 2020-03-08_17-09-39 | 59.17 | [Weights](https://drive.google.com/uc?export=download&id=1D9DyOp9ZiWEILICH6Fq84PQqKMMvLNy8), [Config](https://drive.google.com/uc?export=download&id=1RpyvbTEzt9q6holM7eXDu9VWoPEHmz8V) |
| 2020-03-13_03-46-05 | 59.79 | [Weights](https://drive.google.com/uc?export=download&id=1Sj_Sv8Frq43iMyh3keC-Kbfglk7ES4Lr), [Config](https://drive.google.com/uc?export=download&id=1sb9ZP14WnbCDyNa26bwUtrMbI4sRXSDQ) |
| 2020-03-07_20-22-15 | 59.77 | [Weights](https://drive.google.com/uc?export=download&id=1gLAPcrAzeT71QkZgvFfDMWMD7nI_xfHd), [Config](https://drive.google.com/uc?export=download&id=1VptvP67cRXd7L0c_JsOmc6AqxsGLVWxe) |
| 2020-03-08_07-39-52 | 59.67 | [Weights](https://drive.google.com/uc?export=download&id=15PIn1Mh6O-1f_ImGDJKrAZJ1hVt9dOUg), [Config](https://drive.google.com/uc?export=download&id=1DxhySMut0bFdDBe5vhl981BCHRnmhmTN) |


## Acknowledgments<a name="acknowledgments"/>    

We would like to thank Philippe Weinzaepfel for providing us with the predicted human tubes of their tracking-by-detetion model.










