"""Live visualization helpers for a trained TensorFlow GAN generator.

The animation refreshes generated samples while a restrained particle field
moves behind the image grid. Import this module from the project notebook.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def animate_generator(
    generator: Any,
    latent_dim: int = 100,
    samples: int = 16,
    interval: int = 500,
    seed: int | None = None,
) -> FuncAnimation:
    """Animate fresh generator outputs with a subtle particle backdrop.

    Args:
        generator: A trained TensorFlow/Keras generator model.
        latent_dim: Width of the generator's latent input vector.
        samples: Number of generated images to show (must be a perfect square).
        interval: Delay between frames in milliseconds.
        seed: Optional NumPy random seed for repeatable visualization.

    Returns:
        The Matplotlib animation object. Keep a reference to it in notebooks.
    """
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    if samples <= 0 or int(np.sqrt(samples)) ** 2 != samples:
        raise ValueError("samples must be a positive perfect square")
    if interval <= 0:
        raise ValueError("interval must be positive")

    rng = np.random.default_rng(seed)
    side = int(np.sqrt(samples))
    fig = plt.figure(figsize=(8, 8), facecolor="#0b1220")
    background = fig.add_axes([0, 0, 1, 1], facecolor="#0b1220", zorder=0)
    background.set_xlim(0, 1)
    background.set_ylim(0, 1)
    background.axis("off")

    particle_count = 64
    positions = rng.uniform(0.02, 0.98, size=(particle_count, 2))
    velocities = rng.normal(0, 0.0007, size=(particle_count, 2))
    particles = background.scatter(
        positions[:, 0],
        positions[:, 1],
        s=rng.uniform(3, 10, particle_count),
        c="#7dd3fc",
        alpha=0.28,
        linewidths=0,
    )

    image_axes = []
    image_artists = []
    padding = 0.055
    grid_size = 1 - 2 * padding
    cell = grid_size / side
    for index in range(samples):
        row, column = divmod(index, side)
        left = padding + column * cell + cell * 0.06
        bottom = 1 - padding - (row + 1) * cell + cell * 0.06
        axis = fig.add_axes(
            [left, bottom, cell * 0.88, cell * 0.88],
            facecolor="#111b2e",
            zorder=1,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#26364d")
            spine.set_linewidth(0.8)
        image = axis.imshow(np.zeros((8, 8)), cmap="gray", vmin=0, vmax=1, animated=True)
        image_artists.append(image)
        image_axes.append(axis)

    title = fig.text(
        0.5,
        0.975,
        "LIVE GENERATIONS",
        ha="center",
        va="top",
        color="#dbeafe",
        fontsize=11,
        fontweight="semibold",
        family="sans-serif",
    )

    def update(_frame: int):
        nonlocal positions
        positions += velocities
        for dimension in range(2):
            outside = (positions[:, dimension] < 0.015) | (positions[:, dimension] > 0.985)
            velocities[outside, dimension] *= -1
        positions = np.clip(positions, 0.015, 0.985)
        particles.set_offsets(positions)

        noise = rng.normal(size=(samples, latent_dim)).astype("float32")
        generated = generator(noise, training=False)
        generated = np.asarray(generated)
        for index, image in enumerate(image_artists):
            sample = generated[index]
            if sample.ndim == 3 and sample.shape[-1] == 1:
                sample = sample[..., 0]
            # GAN generators commonly use tanh output in [-1, 1].
            if np.nanmin(sample) < 0:
                sample = (sample + 1) / 2
            image.set_data(np.clip(sample, 0, 1))
        return [particles, title, *image_artists]

    animation = FuncAnimation(
        fig,
        update,
        interval=interval,
        blit=False,
        cache_frame_data=False,
    )
    plt.close(fig)  # Prevent a duplicate static figure in notebook output.
    return animation
