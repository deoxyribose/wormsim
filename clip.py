from jax import lax, vmap
import jax.numpy as jnp
import jax.random as jr

# from celegans import transforms


def convert_to_clip(w, size, R=0.8, eps=0.3, px_spine=0.9):
    """
    Converts the given worms coordinates w of shape:
    (timesteps, num_worms, coord_points, 2) to a collection of 2D frames (clip)
    of shape (size, size).
    Args:
        w: coordinates of the worms.
        size: frame height and width.
        R: radius of the worm.
        eps: Antialiasing smoothing distance(?).
    Returns:
        clip: The simulation of w into consecutive frames.
    """
    w = jnp.round(w).astype(int)
    time, nworms, K, ndim = w.shape

    i = jnp.arange(K)
    r = R * jnp.abs(jnp.sin(jnp.arccos((i - K / 2) / (K / 2 + 0.2))))
    RL = int(3 * R)
    ii, jj = jnp.meshgrid(jnp.arange(-RL, RL + 1), jnp.arange(-RL, RL + 1))

    @vmap
    def draw_circle(r):
        reps = r + eps
        px_value = r / R * px_spine
        d = jnp.sqrt(ii**2 + jj**2)
        return jnp.where(d <= reps, jnp.where(d < r, px_spine, (reps - d) / eps) * px_value, 0)

    circles = draw_circle(r)

    @vmap
    def draw_frame(wt):
        frame = jnp.zeros((size, size), dtype=jnp.float32)

        def place_circle(i, im):
            circle = circles[i % K]
            cx, cy = wt[i]
            current_px = im[cy + jj, cx + ii]
            larger_px = jnp.maximum(circle, current_px)
            negative = ((cy + jj) < 0) | ((cx + ii) < 0)
            new_px = jnp.where(negative, current_px, larger_px)
            im = im.at[cy + jj, cx + ii].set(new_px)
            return im

        frame = lax.fori_loop(0, len(wt), place_circle, init_val=frame)
        return frame

    w_flatten = w.reshape(time, nworms * K, ndim)
    clip = draw_frame(w_flatten)
    return clip