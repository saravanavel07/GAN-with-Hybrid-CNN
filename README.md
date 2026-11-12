# GAN with Hybrid CNN

A Jupyter Notebook project exploring image generation with a Generative Adversarial Network (GAN) and convolutional neural network (CNN) components. The notebook defines a generator and discriminator, training losses and optimizers, a training step, and an image-generation workflow.

## Project contents

- `GAN-With-CNN-HYBRID16.ipynb-main/GAN-With-CNN-HYBRID16.ipynb-main/GAN_With_CNN_HYBRID16.ipynb` — model definition and training notebook.
- `gan_live_animation.py` — optional live visualization helper for generated samples, with a restrained animated particle backdrop.

## Requirements

- Python 3.10 or later
- TensorFlow
- NumPy
- Matplotlib
- Jupyter Notebook or Google Colab

Install the Python dependencies in your environment:

```bash
python -m pip install tensorflow numpy matplotlib notebook
```

## Getting started

1. Clone the repository and open the notebook in Jupyter or upload it to Google Colab.
2. Prepare an image dataset as a NumPy array named `x_train`, with shape `(number_of_images, height, width, channels)` (or `(number_of_images, height, width)` for grayscale images).
3. Normalize image values to the range `[-1, 1]`, and make sure the dataset dimensions match the generator output shape. The current generator is configured for 64 × 64 grayscale images; update the model configuration if your data differs.
4. Run the notebook cells in order to define the models, prepare the dataset, and train the GAN.

> **Dataset note:** The notebook expects `x_train` to be available before the dataset-preparation cell. Add your own dataset-loading and preprocessing steps; no dataset is included in this repository.

## Live generated-sample animation

After training the generator in the notebook, import the helper and display the animation. In a notebook environment, enable JavaScript-based animation output first:

```python
from IPython.display import HTML
from gan_live_animation import animate_generator

animation = animate_generator(
    generator,
    latent_dim=LATENT_DIM,
    samples=16,
    interval=500,
)
HTML(animation.to_jshtml())
```

The visualization periodically samples fresh latent vectors and displays the generated images in a compact grid. A subtle, low-contrast particle field adds motion to the background without obscuring the samples. Keep the `animation` variable alive so the notebook can render the animation.

## Configuration

The notebook currently sets `LATENT_DIM = 100`, `EPOCHS = 50`, and `BATCH_SIZE = 256`. Adjust these values based on your dataset and available compute. Training time and results depend on the input data, hardware, and model configuration.

## Limitations

This repository is an experimental notebook rather than a packaged application. The notebook's training workflow assumes that image data has been loaded and correctly preprocessed beforehand. Validate image dimensions and normalization before training, and use an available GPU where practical.

## License

No license is currently specified. Unless a license is added, standard copyright protections apply and reuse permissions are not explicitly granted.
