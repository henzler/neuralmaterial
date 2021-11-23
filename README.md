# Neural Material

Official code repository for the paper:

**Generative Modelling of BRDF Textures from Flash Images [SIGGRAPH Asia, 2021]**

[Henzler](https://henzler.github.io), [Deschaintre](https://valentin.deschaintre.fr/), [J. Mitra](http://www0.cs.ucl.ac.uk/staff/n.mitra/), [Ritschel](http://www.homepages.ucl.ac.uk/~ucactri/)

**[[Paper](https://arxiv.org/pdf/2102.11861.pdf)] [[Project page](https://henzler.github.io/publication/neuralmaterial/)]**


![Rerendering](images/relighting.gif)

## Data

The dataset is stored under `flash_images` and contains 306 train folders and 116 test folders (including images from [Aitalla et al](https://mediatech.aalto.fi/publications/graphics/TwoShotSVBRDF/)).

### Install dependencies

```
conda env create --name neuralmaterial --file=environment.yml
```

### Training

For training please run
```
python scripts/train.py
```

The default config is located at `config/config_default.yaml`.

### Inference


Note, <test_id> is a relative path in the `flash_images/test` and <weight_name> is exptected to be located in the `trainings/<run_name>` folder.

#### Synthesis

In order to synthesise given flash images located in the `test` folder please run
```
python scripts/test.py --model <model_path> --test_image_id <test_id> --finetune <bool>
```

#### Interpolation

Given two images for interpolation please run

```
python scripts/interpolate.py --model <model_path> --weights1 <weight_name1> --weights2 <weight_name2> --test_image_id1 <test_id1> --test_image_id2 <test_id2>
```

If you would like to use fine-tuned weights please run the `scripts/test.py` command above in order to retrieve them.

#### Examples

Run the file `run_examples.sh` to synthesise / interpolate a few examples.

## Citation

If you find our work useful in your research, please cite:

```
@article{henzler2021neuralmaterial,
  title={Generative Modelling of BRDF Textures from Flash Images},
  author={Henzler, Philipp and Deschaintre, Valentin and Mitra, Niloy J and Ritschel, Tobias},
  journal={ACM Trans Graph (Proc. SIGGRAPH Asia)},
  year={2021},
  volume={40},
  number={6},
}
```

### Contact
If you have any questions, please email Philipp Henzler at philipp.henzler@cs.ucl.ac.uk.