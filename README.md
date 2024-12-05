# GERD: Generated event response data generation

![](https://img.shields.io/badge/DOI-arXiv%3A1408.3644-B31B1B.svg)
![](https://zenodo.org/badge/DOI/10.5281/zenodo.11063678.svg)

This repository lets you generate event-based datasets for objects tracking tasks at arbitrary resolutions subject to arbitrary transformations.
Both the shapes and the transformations can be arbitrarily parameterized, which is useful to study the robustness of event-based algorithms to, for instance, different velocities and transformations&mdash;something that happens *very* frequently in the real world, but that is often overlooked in the literature.

![](affine.gif)

Example render of shapes subject to affine transformation with a relatively high velocity (v=2.56).

## Important simulation details

We apply translation, scaling, rotation, and shearing to the shapes independently.
You can configure the transformations to use different starting conditions and the velocities will be updated according to a specified [PyTorch distribution](https://pytorch.org/docs/stable/distributions.html).
All of this is parameterized in the `RenderParameters` class in the [`render.py`](datasets/render.py) file.

### Activation normalization
Translation velocities are normalized to the pixel grid, meaning that a velocity of 1 in the x axis means that the object moves one pixel to the right every frame.
The other velocities are normalized to produce a similar number of pixel activations, to avoid skewing the dataset towards a specific transformation.

### Fractional velocities and upsampling
A velocity of 0.1 is obviously problematic in a pixel grid, why we use an upsampled grid that, by default, is 8 times the specified resolution.
An event in the downsampled (actual) grid will "trigger" when a certain fraction of the upsampled pixels are turned on.
To accumulate pixel activations in the upsampled grid over time, we use a thresholded integrator.

## Usage

You can install GERD by running 
`pip install git+https://github.com/ncskth/gerd` or by manually pulling the repository and installing the local version with `pip install <path-to-gerd>`.

The code is written in Python using the [PyTorch](https://pytorch.org/) library.
On a low level, we offer a general generating function `render` in the [`render.py`](datasets/render.py) file, that can render specific shapes, defined in [`shapes.py`](datasets/shapes.py).

On a higher level, the [`main.py`](datasets/main.py) file contains a script that generates a dataset of three specific objects moving in a scene: a square, a circle, and a triangle.
We will cover that usecase below:

### 1. Generating data

To generate a dataset, run the `main.py` file (see `python main.py --help` for more information).
The example below generates 1000 videos that translates and scales into the `/data` directory.

```bash
python main.py 1000 /data --translation --scaling --max_velocities 0.1 0.5 1.0
```

Note that the data is saved as a [sparse tensor](https://pytorch.org/docs/stable/sparse.html).

### 2. Using the generated dataset with PyTorch

We provide a PyTorch dataset class in the [`dataset.py`](datasets/dataset.py) file, which is straight-forward to use and only needs the path to the generated dataset as input.
Note that the dataset will output three tensors: a warmup tensor (for use in recurrent networks), an event tensor, and the object positions as labels. 

```python
from datasets.dataset import ShapeDataset

my_dataset = ShapeDataset("/data")
```

By default, the dataset will crop the frames to 40 timesteps and assume that each file contains 128 timesteps. 
You can change this by providing additional parameters to the `ShapeDataset` class.

## Authors

* [Jens E. Pedersen](https://www.kth.se/profile/jeped) (@GitHub [jegp](https://github.com/jegp/)), doctoral student at KTH Royal Institute of Technology, Sweden.
* [Dimitris Korakounis](https://www.kth.se/profile/dimkor), doctoral student at KTH Royal Institute of Technology, Sweden.
* [Raghav Singhal](https://github.com/RaghavSinghal10), visiting student at KTH Royal Institute of Technology, Sweden.

The work has received funding from the EC Horizon 2020 Framework Programme under Grant Agreements 785907 and 945539.

## Citation

If you use this work, please cite it as follows

```
@misc{pedersen2024gerd,
      title={GERD: Geometric event response data generation}, 
      author={Jens Egholm Pedersen and Dimitris Korakovounis and Jörg Conradt},
      year={2024},
      eprint={2412.03259},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.03259}, 
}
```