<div align="center">

# Learnable Earth Parser:
# Discovering 3D Prototypes in Aerial Scans

</div>
 
## Description   
Pytorch implementation of the paper [Learnable Earth Parser: Discovering 3D Prototypes in Aerial Scans, Romain Loiseau, Elliot Vincent, Mathieu Aubry, Loic Landrieu, CVPR 2024](https://romainloiseau.fr/learnable-earth-parser/)

![learnable earth parser](media/learnableearthparser.png)

## Installation

### 1. Create and activate conda environment

```
conda env create -f environment.yml
conda activate learnableearthparser
```

Install the fast superquadrics sampler from [https://github.com/paschalidoud/superquadric_parsing](https://github.com/paschalidoud/superquadric_parsing).

**Optional:** some monitoring routines are implemented with `tensorboard`.

**Note:** this implementation uses [`spconv`](https://github.com/traveller59/spconv) for sparse convolutions, [`hydra`](https://hydra.cc) to manage configuration files and command line arguments, and [`pytorch3d`](https://pytorch3d.org/) to compute the Chamfer distance efficiently.

### 2. Download datasets

Go to the website of desired datasets and download them at `/path/to/datasets/` and use `data.data_dir=/path/to/datasets/dataset-name` as a command line argument for python to find your path when using this implementation.

## How to run

### Training the model

To train our best model, launch :
```bash
python main.py +experiment=xp-name
```

The experiments `crop_field`, `forest`, `greenhouse`, `marina`, `power_plant`, `urban` and `windturbine` are stored in config files in `configs/experiments`. Parameters of the ablations to train AtlasNet-v2 and Superquadrics are stored in `configs/ablations` and can be used by appending `+ablations=ablation-name` to the command line.

### Testing the model

To test the model, launch :
```bash
python main.py \
    +experiment=xp-name \
    mode=test \
    model.load_weights="/path/to/trained/weights.ckpt"
```

Pretrained models can be downloaded [here](https://zenodo.org/record/8276586)

### Citation   

If you use this method and/or this API in your work, please cite our [paper](https://imagine.enpc.fr/~loiseaur/learnable-earth-parser).

```markdown
@article{loiseau2024learnable,
      title={Learnable Earth Parser: Discovering 3D Prototypes in Aerial Scans}, 
      author={Romain Loiseau and Elliot Vincent and Mathieu Aubry and Loic Landrieu},
      journal={CVPR},
      year={2024}
}
```

### Acknowledgements

This work was supported in part by **ANR project READY3D ANR-19-CE23-0007**, ANR under the France 2030 program under the reference **ANR-23-PEIA-0008**, and was granted access to the **HPC resources of IDRIS** under the allocation 2022-AD011012096R2 made by GENCI. The work of MA was partly supported by the **European Research Council (ERC project DISCOVER, number 101076028)**. The scenes of Earth Parser Dataset were acquired and annotated by the **[LiDAR-HD](https://geoservices.ign.fr/lidarhd)** project. We thank **Zenodo** for hosting the dataset. We thank Zeynep Sonat Baltaci, Emile Blettery, Nicolas Dufour, Antoine Guedon, Helen Mair Rawsthorne, Tom Monnier, Damien Robert, Mathis Petrovich and Yannis Siglidis for inspiring discussions and valuable feedback.