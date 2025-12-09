"""Matplotlib tracking code injected into E2B sandbox.

This code is executed in the E2B sandbox to track chart creation via savefig,
close, and show calls. It extracts chart data before figures are closed.
"""

TRACKING_CODE = """
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import json

if not hasattr(plt, '_jetflow_tracking_installed'):
    _original_savefig = plt.Figure.savefig
    _original_close = plt.close
    _original_show = plt.show
    plt._jetflow_tracking_installed = True

_jetflow_pending_charts = []


def _color_to_hex(color):
    '''Convert matplotlib color to hex string.'''
    try:
        if color is None:
            return None
        if isinstance(color, str):
            if color.startswith('#'):
                return color
            return mcolors.to_hex(color)
        if hasattr(color, '__iter__') and len(color) >= 3:
            return mcolors.to_hex(color[:3])
    except:
        pass
    return None


def _linestyle_to_name(ls):
    '''Convert linestyle to readable name.'''
    mapping = {
        '-': 'solid', 'solid': 'solid',
        '--': 'dashed', 'dashed': 'dashed',
        ':': 'dotted', 'dotted': 'dotted',
        '-.': 'dashdot', 'dashdot': 'dashdot',
    }
    return mapping.get(ls, None)


def _tracked_savefig(self, fname, *args, **kwargs):
    filename = os.path.basename(str(fname))
    if '.' in filename:
        filename = filename.rsplit('.', 1)[0]
    self._jetflow_chart_id = filename
    return _original_savefig(self, fname, *args, **kwargs)


def _get_bar_labels(ax):
    labels = []
    try:
        from matplotlib.container import BarContainer
        for container in ax.containers:
            if isinstance(container, BarContainer):
                label = container.get_label()
                if label and not label.startswith('_'):
                    labels.append(label)
    except:
        pass
    return labels


def _extract_line_data(line):
    '''Extract data and styling from a Line2D object.'''
    xdata, ydata = line.get_xdata(), line.get_ydata()
    return {
        'x': xdata.tolist() if hasattr(xdata, 'tolist') else list(xdata),
        'y': ydata.tolist() if hasattr(ydata, 'tolist') else list(ydata),
        'label': line.get_label(),
        'color': _color_to_hex(line.get_color()),
        'linestyle': _linestyle_to_name(line.get_linestyle()),
        'linewidth': line.get_linewidth(),
        'marker': line.get_marker() if line.get_marker() != 'None' else None,
        'markersize': line.get_markersize() if line.get_marker() != 'None' else None,
        'alpha': line.get_alpha(),
    }


def _extract_patch_data(patch):
    '''Extract data and styling from a patch (bar).'''
    return {
        'x': patch.get_x(),
        'y': patch.get_y(),
        'width': patch.get_width(),
        'height': patch.get_height(),
        'color': _color_to_hex(patch.get_facecolor()),
        'alpha': patch.get_alpha(),
    }


def _extract_collection_data(coll):
    '''Extract data and styling from a PathCollection (scatter).'''
    offsets = coll.get_offsets()
    if len(offsets) == 0:
        return None

    # Get color - could be array or single value
    facecolors = coll.get_facecolors()
    color = None
    if len(facecolors) > 0:
        color = _color_to_hex(facecolors[0])

    return {
        'x': offsets[:, 0].tolist() if hasattr(offsets[:, 0], 'tolist') else list(offsets[:, 0]),
        'y': offsets[:, 1].tolist() if hasattr(offsets[:, 1], 'tolist') else list(offsets[:, 1]),
        'color': color,
        'alpha': coll.get_alpha(),
    }


def _extract_pie_data(ax):
    '''Extract pie chart data from wedge patches.'''
    from matplotlib.patches import Wedge

    wedges = [p for p in ax.patches if isinstance(p, Wedge)]
    if not wedges:
        return None

    # Get text labels - pie charts have text artists for labels
    texts = ax.texts
    labels = [t.get_text() for t in texts if t.get_text().strip()]

    # Extract wedge data
    slices = []
    for i, wedge in enumerate(wedges):
        theta1, theta2 = wedge.theta1, wedge.theta2
        # Value is proportional to angle
        value = (theta2 - theta1) / 360.0

        label = labels[i] if i < len(labels) else f'slice-{i+1}'

        # Check for explode (center offset from origin)
        center = wedge.center
        explode = (center[0]**2 + center[1]**2)**0.5 if center != (0, 0) else 0.0

        slices.append({
            'label': label,
            'value': value,
            'color': _color_to_hex(wedge.get_facecolor()),
            'explode': round(explode, 4),
        })

    return slices


def _extract_area_data(ax):
    '''Extract area chart data from PolyCollections (fill_between).'''
    from matplotlib.collections import PolyCollection

    areas = []
    for coll in ax.collections:
        if isinstance(coll, PolyCollection):
            paths = coll.get_paths()
            if not paths:
                continue

            # Extract vertices from the polygon
            # fill_between creates a polygon with bottom and top boundaries
            verts = paths[0].vertices
            if len(verts) < 4:
                continue

            # The polygon goes: bottom-left -> top points -> bottom-right -> bottom points (reversed)
            # We need to extract just the x and y values
            n = len(verts) // 2
            x_vals = verts[:n, 0].tolist()
            y_vals = verts[:n, 1].tolist()

            areas.append({
                'x': x_vals,
                'y': y_vals,
                'label': coll.get_label() if hasattr(coll, 'get_label') else None,
                'color': _color_to_hex(coll.get_facecolor()[0]) if len(coll.get_facecolor()) > 0 else None,
                'alpha': coll.get_alpha(),
            })

    return areas if areas else None


def _extract_figure_data(fig):
    fig_label = fig.get_label() or None
    saved_filename = getattr(fig, '_jetflow_chart_id', None)
    subtitle = getattr(fig, 'subtitle', None)
    data_source = getattr(fig, 'data_source', None)
    citations = getattr(fig, 'citations', [])

    axes_data = []
    for ax_idx, ax in enumerate(fig.get_axes()):
        shared_x_ids = [id(other) for other in fig.get_axes() if other != ax and ax.get_shared_x_axes().joined(ax, other)]

        axis_data = {
            'fig_num': fig.number,
            'ax_idx': ax_idx,
            'ax_id': id(ax),
            'fig_label': fig_label,
            'saved_filename': saved_filename,
            'var_name': None,
            'subtitle': subtitle,
            'data_source': data_source,
            'citations': citations,
            'title': ax.get_title() or None,
            'xlabel': ax.get_xlabel() or None,
            'ylabel': ax.get_ylabel() or None,
            'xscale': ax.get_xscale(),
            'yscale': ax.get_yscale(),
            'shared_x_ids': shared_x_ids,
            'lines': [],
            'patches': [],
            'collections': [],
            'pie_slices': None,
            'area_fills': None,
            'bar_labels': _get_bar_labels(ax),
            'xtick_labels': [t.get_text() for t in ax.get_xticklabels()],
            'ytick_labels': [t.get_text() for t in ax.get_yticklabels()]
        }

        # Extract lines with styling
        for line in ax.get_lines():
            axis_data['lines'].append(_extract_line_data(line))

        # Extract patches (bars) with styling
        for patch in ax.patches:
            # Skip Wedge patches (handled separately for pie)
            from matplotlib.patches import Wedge
            if not isinstance(patch, Wedge):
                axis_data['patches'].append(_extract_patch_data(patch))

        # Extract collections (scatter) with styling
        from matplotlib.collections import PolyCollection
        for coll in ax.collections:
            # Skip PolyCollections (area charts)
            if isinstance(coll, PolyCollection):
                continue
            data = _extract_collection_data(coll)
            if data:
                axis_data['collections'].append(data)

        # Extract pie chart data
        pie_data = _extract_pie_data(ax)
        if pie_data:
            axis_data['pie_slices'] = pie_data

        # Extract area chart data
        area_data = _extract_area_data(ax)
        if area_data:
            axis_data['area_fills'] = area_data

        axes_data.append(axis_data)

    return axes_data


def _tracked_close(fig=None):
    global _jetflow_pending_charts
    if fig is None:
        for fig_num in plt.get_fignums():
            _jetflow_pending_charts.extend(_extract_figure_data(plt.figure(fig_num)))
    elif isinstance(fig, plt.Figure):
        _jetflow_pending_charts.extend(_extract_figure_data(fig))
    elif isinstance(fig, int):
        if fig in plt.get_fignums():
            _jetflow_pending_charts.extend(_extract_figure_data(plt.figure(fig)))
    elif fig == 'all':
        for fig_num in plt.get_fignums():
            _jetflow_pending_charts.extend(_extract_figure_data(plt.figure(fig_num)))
    return _original_close(fig)


def _tracked_show(*args, **kwargs):
    global _jetflow_pending_charts
    figs_to_close = list(plt.get_fignums())
    for fig_num in figs_to_close:
        _jetflow_pending_charts.extend(_extract_figure_data(plt.figure(fig_num)))
    result = _original_show(*args, **kwargs)
    for fig_num in figs_to_close:
        try:
            _original_close(fig_num)
        except:
            pass
    return result


if not hasattr(plt.Figure, '_jetflow_savefig_patched'):
    plt.Figure.savefig = _tracked_savefig
    plt.Figure._jetflow_savefig_patched = True
if not hasattr(plt, '_jetflow_close_patched'):
    plt.close = _tracked_close
    plt._jetflow_close_patched = True
if not hasattr(plt, '_jetflow_show_patched'):
    plt.show = _tracked_show
    plt._jetflow_show_patched = True
"""
