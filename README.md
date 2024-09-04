# *Uv*-free Mesh Texture Generation with Denoising and Heat Diffusion

This repository provides the official implementation of UV3-TeD:

### [UV-free Texture Generation with Denoising and Geodesic Heat Diffusions](https://arxiv.org/abs/2408.16762)
[Simone Foti](https://www.simofoti.com/), [Stefanos Zafeiriou](https://scholar.google.com/citations?user=QKOH5iYAAAAJ&hl=en), [Tolga Birdal](http://tolgabirdal.github.io/)\
Imperial College London


<video src="https://github.com/simofoti/UV3-TeD/blob/main/assets/teaser_video.mp4" width="320" height="240" controls></video>


## Installation

We suggest creating a mamba environment, but conda can be used as well by simply 
replacing `mamba` with `conda`.

To create the environment, open a terminal and type:
```bash
mamba create -n uv3-ted
```

Then activate the environment with:
```bash
mamba activate uv3-ted
```

Then run the the following commands to install the necessary dependencies:
```bash
mamba install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install pyg -c pyg
mamba install pytorch-scatter pytorch-cluster pytorch-sparse -c pyg

pip install diffusers["torch"]
pip install mitsuba

pip install trimesh Pillow rtree
pip install "pyglet<2"
pip install scipy robust_laplacian polyscope pandas point-cloud-utils
pip install func_timeout tb-nightly npyvista
```

If you want to evaluate the performance of the model run also the following:

```bash
pip install clean-fid lpips
```

## Datasets

Installation instructions should be automatically printed when launching the 
code if data are not found or automatic download is not implemented. 

Permissions to download the data may be required. Please, refer to the 
[ShapeNet](https://huggingface.co/datasets/ShapeNet/ShapeNetCore) and 
[Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html#download) 
dataset websites for more information.

 
## Prepare Your Configuration File
 
We made available a configuration file for each experiment. Make sure 
the paths in the config file are correct. In particular, you might have to 
change `root` according to where the data were downloaded.
 

## Train and Test
 
After cloning the repo open a terminal and go to the project directory. 
Ensure that your mamba/conda environment is active.

To start the training from the project repo simply run:
```bash
python train.py --config=configs/<A_CONFIG_FILE>.yaml --id=<NAME_OF_YOUR_EXPERIMENT>
```

Basic tests will automatically run on the validation set at the end of the 
training. If you wish to run experiment on the test set or to run other 
experiments you can uncomment any function call 
at the end of `test.py`. If your model has alredy been trained or you are using 
our pretrained model, you can run tests without training:

```bash
python test.py --id=<NAME_OF_YOUR_EXPERIMENT>
```
Note that NAME_OF_YOUR_EXPERIMENT is also the name of the folder containing the
pretrained model.

The following parameters can also be used:
- `--output_path=<PATH>`: path to where outputs are going to be stored.
- `--processed_dir_name=<PATH>`: relative path to where all the preprocessed 
    files are going to be stored. This path is relative to the folder where your 
    data are stored. 
- `--resume`: resume the training (available only when launching *train.py*).
- `--profile`: run a few training steps to profile model performance
    (available only when launching *train.py*).
- `--batch_size=<n>`: overrides the batch size specified in the config file,
    (available only when launching *test.py*).

## Run LPIPS
Lpips can be used launching a simple script from the LPIPS library. 
After running the tests follow these steps:

Clone the LPIPS repo and cd into it:
```bash
git clone https://github.com/richzhang/PerceptualSimilarity.git
cd ./PerceptualSimilarity
```
Then run:
```bash
python lpips_2dirs.py -d0 <PATH_TO_A_DIR_CONTAINING_A_SET_OF_RENDERED_SHAPES> -d1 <PATH_TO_ANOTHER_DIR_CONTAINING_A_SET_OF_RENDERED_SHAPES> -o <PATH_TO_OUT_TXT_FILE> --use_gpu
```

