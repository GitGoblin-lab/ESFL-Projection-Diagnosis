
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy.ndimage import gaussian_filter

def generate_rugged_landscape(max_x=1000, max_y=500,
                              max_z=75, resolution=150,
                              seed=42):

    rng = np.random.default_rng(seed)

    x = np.linspace(0, max_x, resolution)
    y = np.linspace(0, max_y, resolution)
    X, Y = np.meshgrid(x, y)


    xn = X / max_x
    yn = Y / max_y


    # -------------------------------------------------
    trend = 0.53 * xn + 0.47 * yn


    trend += (
        0.09 * np.sin(2.5 * np.pi * xn + 0.6)
        + 0.07 * np.cos(2.0 * np.pi * yn + 0.8)
        + 0.04 * np.sin(2 * np.pi * (xn + yn))
    )


    # -------------------------------------------------
    noise_low = gaussian_filter(
        rng.standard_normal((resolution, resolution)),
        sigma=12
    )
    noise_low /= np.std(noise_low)


    # -------------------------------------------------
    noise_mid = gaussian_filter(
        rng.standard_normal((resolution, resolution)),
        sigma=3.5
    )
    noise_mid /= np.std(noise_mid)


    # -------------------------------------------------
    noise_high = gaussian_filter(
        rng.standard_normal((resolution, resolution)),
        sigma=0.9
    )
    noise_high /= np.std(noise_high)


    # -------------------------------------------------
    Z = (
        0.68 * trend
        + 0.105 * noise_low
        + 0.065 * noise_mid
        + 0.032 * noise_high
    )


    # -------------------------------------------------
    for _ in range(28):

        cx = rng.uniform(0.05, 0.95)
        cy = rng.uniform(0.05, 0.95)

        amplitude = rng.uniform(-0.13, 0.18)

        width_x = rng.uniform(0.012, 0.055)
        width_y = rng.uniform(0.012, 0.065)

        peak = amplitude * np.exp(
            -(
                ((xn - cx) / width_x) ** 2
                + ((yn - cy) / width_y) ** 2
            ) / 2
        )

        Z += peak


    # -------------------------------------------------
    Z = Z + 0.12 * np.tanh(4 * (Z - 0.5))


    Z = (Z - Z.min()) / (Z.max() - Z.min())
    Z = Z * max_z

    return X, Y, Z



# =====================================================
def plot_rugged_landscape(X, Y, Z):

    fig = plt.figure(figsize=(7.2, 7.2), dpi=300)

    ax = fig.add_subplot(111, projection='3d')

    cmap = cm.get_cmap('Spectral_r')

    norm = colors.Normalize(vmin=4.50, vmax=89.48)

    color_values = 4.50 + Z / 75.0 * (89.48 - 4.50)

    facecolors = cmap(norm(color_values))

    surf = ax.plot_surface(
        X, Y, Z,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0.12,
        edgecolor=(0.25, 0.25, 0.25, 0.18),
        antialiased=True,
        shade=False
    )

    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 500)
    ax.set_zlim(0, 75)

    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.set_yticks([0, 100, 200, 300, 400, 500])
    ax.set_zticks([0, 13, 26, 39, 52, 65])

    ax.set_xlabel('X', labelpad=8)
    ax.set_ylabel('Y', labelpad=8)
    ax.set_zlabel('')

    ax.view_init(elev=35, azim=-140)

    ax.set_box_aspect((1.0, 0.65, 0.65))

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((0.96, 0.96, 0.96, 1))
        axis.pane.set_edgecolor((0.85, 0.85, 0.85, 1))

        axis._axinfo["grid"]["linestyle"] = ':'
        axis._axinfo["grid"]["linewidth"] = 0.6
        axis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.5)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        shrink=0.58,
        aspect=18,
        pad=0.04
    )

    cbar.set_ticks([
        4.50, 16.64, 28.78, 40.92,
        53.06, 65.20, 77.34, 89.48
    ])

    cbar.ax.tick_params(labelsize=9)

    plt.subplots_adjust(
        left=0.02,
        right=0.94,
        bottom=0.02,
        top=0.98
    )

    plt.show()



# =====================================================
X, Y, Z = generate_rugged_landscape(
    max_x=1000,
    max_y=500,
    max_z=75,
    resolution=150,
    seed=42
)

plot_rugged_landscape(X, Y, Z)
