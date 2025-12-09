"""Chart extraction from E2B matplotlib figures."""

import json
from typing import List, Dict, Any, Set, TYPE_CHECKING

from jetflow.actions.chart_utils import group_axes_by_twins
from jetflow.actions.chart_processing import ChartProcessor

if TYPE_CHECKING:
    from jetflow.models.chart import Chart


_FIGURE_HASH_CODE = """
import json
import matplotlib.pyplot as plt
import hashlib

def hash_figure(fig):
    try:
        props = {'num_axes': len(fig.axes), 'axes_data': []}
        for ax in fig.axes:
            ax_data = {
                'title': ax.get_title(), 'xlabel': ax.get_xlabel(), 'ylabel': ax.get_ylabel(),
                'num_lines': len(ax.get_lines()), 'num_patches': len(ax.patches),
                'num_collections': len(ax.collections), 'xlim': ax.get_xlim(), 'ylim': ax.get_ylim(),
            }
            line_hashes = []
            for line in ax.get_lines():
                xdata, ydata = line.get_xdata(), line.get_ydata()
                line_hashes.append(hashlib.md5(f"{xdata.tobytes()}{ydata.tobytes()}".encode()).hexdigest()[:8])
            if line_hashes:
                ax_data['line_hashes'] = line_hashes
            if ax.patches:
                patch_data = [f"{p.get_x()},{p.get_height()},{p.get_width()}" for p in ax.patches]
                ax_data['patch_hash'] = hashlib.md5(''.join(patch_data).encode()).hexdigest()[:8]
            collection_hashes = []
            for coll in ax.collections:
                try:
                    offsets = coll.get_offsets()
                    if len(offsets) > 0:
                        collection_hashes.append(hashlib.md5(offsets.tobytes()).hexdigest()[:8])
                except: pass
            if collection_hashes:
                ax_data['collection_hashes'] = collection_hashes
            props['axes_data'].append(ax_data)
        return hashlib.md5(json.dumps(props, sort_keys=True, default=str).encode()).hexdigest()
    except Exception as e:
        return f"error:{str(e)}"

result = {}
for fig_num in plt.get_fignums():
    result[str(fig_num)] = hash_figure(plt.figure(fig_num))
print(json.dumps(result))
"""

_RAW_DATA_EXTRACTION = """
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def _color_to_hex(color):
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
    mapping = {
        '-': 'solid', 'solid': 'solid',
        '--': 'dashed', 'dashed': 'dashed',
        ':': 'dotted', 'dotted': 'dotted',
        '-.': 'dashdot', 'dashdot': 'dashdot',
    }
    return mapping.get(ls, None)

def get_bar_labels(ax):
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
    if not labels:
        try:
            from matplotlib.container import BarContainer
            handles, lbls = ax.get_legend_handles_labels()
            for handle, label in zip(handles, lbls):
                if isinstance(handle, BarContainer):
                    labels.append(label)
        except:
            pass
    return labels

def extract_pie_data(ax):
    from matplotlib.patches import Wedge
    wedges = [p for p in ax.patches if isinstance(p, Wedge)]
    if not wedges:
        return None
    texts = ax.texts
    labels = [t.get_text() for t in texts if t.get_text().strip()]
    slices = []
    for i, wedge in enumerate(wedges):
        theta1, theta2 = wedge.theta1, wedge.theta2
        value = (theta2 - theta1) / 360.0
        label = labels[i] if i < len(labels) else f'slice-{i+1}'
        center = wedge.center
        explode = (center[0]**2 + center[1]**2)**0.5 if center != (0, 0) else 0.0
        slices.append({
            'label': label,
            'value': value,
            'color': _color_to_hex(wedge.get_facecolor()),
            'explode': round(explode, 4),
        })
    return slices

def extract_area_data(ax):
    from matplotlib.collections import PolyCollection
    areas = []
    for coll in ax.collections:
        if isinstance(coll, PolyCollection):
            paths = coll.get_paths()
            if not paths:
                continue
            verts = paths[0].vertices
            if len(verts) < 4:
                continue
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

def dump_raw_axes():
    raw_axes = []
    fig_var_names = {}
    try:
        for var_name, var_value in list(globals().items()):
            if isinstance(var_value, plt.Figure) and not var_name.startswith('_'):
                fig_var_names[var_value.number] = var_name
    except:
        pass

    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        fig_label = fig.get_label() or None
        saved_filename = getattr(fig, '_jetflow_chart_id', None)
        var_name = fig_var_names.get(fig_num)
        subtitle = getattr(fig, 'subtitle', None)
        data_source = getattr(fig, 'data_source', None)
        citations = getattr(fig, 'citations', [])

        for ax_idx, ax in enumerate(fig.get_axes()):
            shared_x_ids = [id(other) for other in fig.get_axes() if other != ax and ax.get_shared_x_axes().joined(ax, other)]
            axis_data = {
                'fig_num': fig_num, 'ax_idx': ax_idx, 'ax_id': id(ax),
                'fig_label': fig_label, 'saved_filename': saved_filename, 'var_name': var_name,
                'subtitle': subtitle, 'data_source': data_source, 'citations': citations,
                'title': ax.get_title() or None, 'xlabel': ax.get_xlabel() or None, 'ylabel': ax.get_ylabel() or None,
                'xscale': ax.get_xscale(), 'yscale': ax.get_yscale(), 'shared_x_ids': shared_x_ids,
                'lines': [], 'patches': [], 'collections': [],
                'pie_slices': None, 'area_fills': None,
                'bar_labels': get_bar_labels(ax),
                'xtick_labels': [t.get_text() for t in ax.get_xticklabels()],
                'ytick_labels': [t.get_text() for t in ax.get_yticklabels()]
            }

            # Extract lines with styling
            for line in ax.get_lines():
                xdata, ydata = line.get_xdata(), line.get_ydata()
                axis_data['lines'].append({
                    'x': xdata.tolist() if hasattr(xdata, 'tolist') else list(xdata),
                    'y': ydata.tolist() if hasattr(ydata, 'tolist') else list(ydata),
                    'label': line.get_label(),
                    'color': _color_to_hex(line.get_color()),
                    'linestyle': _linestyle_to_name(line.get_linestyle()),
                    'linewidth': line.get_linewidth(),
                    'marker': line.get_marker() if line.get_marker() != 'None' else None,
                    'markersize': line.get_markersize() if line.get_marker() != 'None' else None,
                    'alpha': line.get_alpha(),
                })

            # Extract patches (bars) with styling - skip Wedges for pie
            from matplotlib.patches import Wedge
            for patch in ax.patches:
                if not isinstance(patch, Wedge):
                    axis_data['patches'].append({
                        'x': patch.get_x(), 'y': patch.get_y(),
                        'width': patch.get_width(), 'height': patch.get_height(),
                        'color': _color_to_hex(patch.get_facecolor()),
                        'alpha': patch.get_alpha(),
                    })

            # Extract collections (scatter) with styling - skip PolyCollections for area
            from matplotlib.collections import PolyCollection
            for coll in ax.collections:
                if isinstance(coll, PolyCollection):
                    continue
                offsets = coll.get_offsets()
                if len(offsets) > 0:
                    facecolors = coll.get_facecolors()
                    color = _color_to_hex(facecolors[0]) if len(facecolors) > 0 else None
                    axis_data['collections'].append({
                        'x': offsets[:, 0].tolist() if hasattr(offsets[:, 0], 'tolist') else list(offsets[:, 0]),
                        'y': offsets[:, 1].tolist() if hasattr(offsets[:, 0], 'tolist') else list(offsets[:, 1]),
                        'color': color,
                        'alpha': coll.get_alpha(),
                    })

            # Extract pie data
            pie_data = extract_pie_data(ax)
            if pie_data:
                axis_data['pie_slices'] = pie_data

            # Extract area data
            area_data = extract_area_data(ax)
            if area_data:
                axis_data['area_fills'] = area_data

            raw_axes.append(axis_data)
    return raw_axes

try:
    print(json.dumps(dump_raw_axes(), default=str))
except Exception as e:
    print(json.dumps({'error': str(e)}))
"""


class E2BChartExtractor:
    """Extracts chart data from E2B sandbox matplotlib figures."""

    def __init__(self, executor):
        self._executor = executor

    def get_figure_hashes(self) -> Dict[str, str]:
        """Get hash fingerprints of all current matplotlib figures."""
        try:
            result = self._executor.run_code(_FIGURE_HASH_CODE)
            if result.logs and result.logs.stdout:
                output = "\n".join(result.logs.stdout).strip()
                return json.loads(output)
        except:
            pass
        return {}

    def get_new_figures(self, pre_hashes: Dict[str, str]) -> Set[str]:
        """Get figure numbers that are new or modified since pre_hashes."""
        post_hashes = self.get_figure_hashes()
        return {
            fig_num for fig_num, post_hash in post_hashes.items()
            if pre_hashes.get(fig_num) != post_hash
        }

    def close_figures(self, fig_nums: Set[str]) -> None:
        """Close specified figures."""
        if not fig_nums:
            return
        try:
            code = f"import matplotlib.pyplot as plt\nfor n in [{','.join(fig_nums)}]:\n    try: plt.close(n)\n    except: pass"
            self._executor.run_code(code)
        except:
            pass

    def extract(self, fig_nums: Set[str] = None) -> List['Chart']:
        """Extract charts from E2B sandbox."""
        raw_axes = self._fetch_raw_axes()
        if not raw_axes:
            return []

        if fig_nums is not None:
            target_nums = {int(n) for n in fig_nums}
            raw_axes = [ax for ax in raw_axes if ax['fig_num'] in target_nums]
            if not raw_axes:
                return []

        axis_groups = group_axes_by_twins(raw_axes)
        return [c for c in (ChartProcessor(g).process() for g in axis_groups) if c]

    def _fetch_raw_axes(self) -> List[Dict[str, Any]]:
        """Execute extraction code in E2B and return raw axis data."""
        try:
            result = self._executor.run_code(_RAW_DATA_EXTRACTION)
            if not result.logs or not result.logs.stdout:
                return []
            output = "\n".join(result.logs.stdout).strip()
            data = json.loads(output)
            if isinstance(data, dict) and 'error' in data:
                return []
            return data if isinstance(data, list) else []
        except:
            return []
