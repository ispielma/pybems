"""Plotting helpers for propagation results."""

from __future__ import annotations

import numpy as np

from .elements import AnnularAperture, AntiAperture, CircularAperture
from .system import PropagationResult


_APERTURE_TYPES = (CircularAperture, AntiAperture, AnnularAperture)


def _blocked_segments(element, transverse_limit: float):
    """Return signed transverse intervals occupied by an element's mask."""

    if isinstance(element, CircularAperture):
        radius = min(element.radius, transverse_limit)
        if radius >= transverse_limit:
            return ()
        return ((-transverse_limit, -radius), (radius, transverse_limit))

    if isinstance(element, AntiAperture):
        radius = min(element.radius, transverse_limit)
        return ((-radius, radius),)

    if isinstance(element, AnnularAperture):
        inner = min(element.inner_radius, transverse_limit)
        outer = min(element.outer_radius, transverse_limit)
        segments = []
        if inner > 0:
            segments.append((-inner, inner))
        if outer < transverse_limit:
            segments.extend(
                ((-transverse_limit, -outer), (outer, transverse_limit))
            )
        return tuple(segments)

    clear_radius = getattr(element, "clear_radius", np.inf)
    if np.isfinite(clear_radius) and clear_radius < transverse_limit:
        return (
            (-transverse_limit, -clear_radius),
            (clear_radius, transverse_limit),
        )
    return ()


def plot_propagation(
    result: PropagationResult,
    *,
    logarithmic: bool = True,
    floor: float = 1e-6,
    length_scale: float = 1e-3,
    length_unit: str = "mm",
):
    """Plot RMS width above a signed-diameter intensity map.

    The colorbar occupies its own GridSpec column, so the axial axes of the two
    data panels remain aligned.
    """

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if floor <= 0:
        raise ValueError("floor must be positive")
    if length_scale <= 0:
        raise ValueError("length_scale must be positive")

    x, field = result.diameter_view()
    intensity = np.abs(field) ** 2
    maximum = float(np.max(intensity))
    normalized = intensity / maximum if maximum > 0 else intensity

    figure = plt.figure(figsize=(10, 7), constrained_layout=True)
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1, 0.035),
        height_ratios=(1, 3),
    )
    width_axis = figure.add_subplot(grid[0, 0])
    field_axis = figure.add_subplot(grid[1, 0], sharex=width_axis)
    color_axis = figure.add_subplot(grid[1, 1])

    z_plot = result.z / length_scale
    width_axis.plot(z_plot, result.rms_x / length_scale)
    width_axis.set_ylabel(f"RMS width ({length_unit})")
    width_axis.grid(alpha=0.25)

    if logarithmic:
        image = field_axis.pcolormesh(
            z_plot,
            x / length_scale,
            np.maximum(normalized, floor),
            shading="auto",
            norm=LogNorm(vmin=floor, vmax=1),
            cmap="inferno",
        )
        color_label = "Normalized intensity (log scale)"
    else:
        image = field_axis.pcolormesh(
            z_plot,
            x / length_scale,
            normalized,
            shading="auto",
            vmin=0,
            vmax=1,
            cmap="inferno",
        )
        color_label = "Normalized intensity"

    transverse_limit = float(np.max(x))
    for location in result.elements:
        position = location.z / length_scale
        width_axis.axvline(position, color="white", alpha=0.35, linewidth=0.8)
        if not isinstance(location.element, _APERTURE_TYPES):
            field_axis.axvline(
                position,
                color="cyan",
                alpha=0.55,
                linewidth=0.8,
            )

        segments = _blocked_segments(location.element, transverse_limit)
        if segments:
            field_axis.vlines(
                [position] * len(segments),
                [lower / length_scale for lower, _ in segments],
                [upper / length_scale for _, upper in segments],
                color="cyan",
                alpha=0.85,
                linewidth=2.0,
            )

    field_axis.set_xlabel(f"Axial position ({length_unit})")
    field_axis.set_ylabel(f"Transverse position ({length_unit})")
    figure.colorbar(image, cax=color_axis, label=color_label)
    return figure, (width_axis, field_axis, color_axis)
