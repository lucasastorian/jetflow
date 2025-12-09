"""Matplotlib tracking code injected into E2B sandbox.

This code is executed in the E2B sandbox to track chart creation via savefig,
close, and show calls. It extracts chart data before figures are closed.
"""

TRACKING_CODE = """
import matplotlib.pyplot as plt
import os
import json

if not hasattr(plt, '_jetflow_tracking_installed'):
    _original_savefig = plt.Figure.savefig
    _original_close = plt.close
    _original_show = plt.show
    plt._jetflow_tracking_installed = True

_jetflow_pending_charts = []

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
            'fig_num': fig.number, 'ax_idx': ax_idx, 'ax_id': id(ax),
            'fig_label': fig_label, 'saved_filename': saved_filename, 'var_name': None,
            'subtitle': subtitle, 'data_source': data_source, 'citations': citations,
            'title': ax.get_title() or None, 'xlabel': ax.get_xlabel() or None, 'ylabel': ax.get_ylabel() or None,
            'xscale': ax.get_xscale(), 'yscale': ax.get_yscale(), 'shared_x_ids': shared_x_ids,
            'lines': [], 'patches': [], 'collections': [],
            'bar_labels': _get_bar_labels(ax),
            'xtick_labels': [t.get_text() for t in ax.get_xticklabels()],
            'ytick_labels': [t.get_text() for t in ax.get_yticklabels()]
        }
        for line in ax.get_lines():
            xdata, ydata = line.get_xdata(), line.get_ydata()
            axis_data['lines'].append({
                'x': xdata.tolist() if hasattr(xdata, 'tolist') else list(xdata),
                'y': ydata.tolist() if hasattr(ydata, 'tolist') else list(ydata),
                'label': line.get_label()
            })
        for patch in ax.patches:
            axis_data['patches'].append({
                'x': patch.get_x(), 'y': patch.get_y(),
                'width': patch.get_width(), 'height': patch.get_height()
            })
        for coll in ax.collections:
            offsets = coll.get_offsets()
            if len(offsets) > 0:
                axis_data['collections'].append({
                    'x': offsets[:, 0].tolist() if hasattr(offsets[:, 0], 'tolist') else list(offsets[:, 0]),
                    'y': offsets[:, 1].tolist() if hasattr(offsets[:, 0], 'tolist') else list(offsets[:, 1])
                })
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
