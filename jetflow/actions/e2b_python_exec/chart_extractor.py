"""Chart extraction from E2B matplotlib figures."""

import json
from typing import List, Dict, Any, Set, TYPE_CHECKING

from jetflow.actions.chart_processing import group_axes_by_twins, process_axis_group_to_chart

if TYPE_CHECKING:
    from jetflow.models.chart import Chart


FIGURE_HASH_CODE = """
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

RAW_DATA_EXTRACTION = """
import json
import matplotlib.pyplot as plt

def dump_raw_axes():
    raw_axes = []

    # Try to find variable names for figures by inspecting globals
    fig_var_names = {}
    try:
        for var_name, var_value in list(globals().items()):
            if isinstance(var_value, plt.Figure) and not var_name.startswith('_'):
                fig_var_names[var_value.number] = var_name
    except:
        pass

    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)

        # Try to get custom chart ID from various sources
        fig_label = fig.get_label() or None
        saved_filename = getattr(fig, '_jetflow_chart_id', None)
        var_name = fig_var_names.get(fig_num)

        # Extract LLM-attached metadata
        subtitle = getattr(fig, 'jetflow_subtitle', None)
        data_source = getattr(fig, 'jetflow_data_source', None)
        citations = getattr(fig, 'jetflow_citations', [])

        for ax_idx, ax in enumerate(fig.get_axes()):
            shared_x_ids = [id(other) for other in fig.get_axes() if other != ax and ax.get_shared_x_axes().joined(ax, other)]
            axis_data = {
                'fig_num': fig_num, 'ax_idx': ax_idx, 'ax_id': id(ax),
                'fig_label': fig_label,
                'saved_filename': saved_filename,
                'var_name': var_name,
                'subtitle': subtitle,
                'data_source': data_source,
                'citations': citations,
                'title': ax.get_title() or None, 'xlabel': ax.get_xlabel() or None, 'ylabel': ax.get_ylabel() or None,
                'xscale': ax.get_xscale(), 'yscale': ax.get_yscale(), 'shared_x_ids': shared_x_ids,
                'lines': [], 'patches': [], 'collections': [],
                'xtick_labels': [t.get_text() for t in ax.get_xticklabels()]
            }
            for line in ax.get_lines():
                xdata, ydata = line.get_xdata(), line.get_ydata()
                axis_data['lines'].append({
                    'x': xdata.tolist() if hasattr(xdata, 'tolist') else list(xdata),
                    'y': ydata.tolist() if hasattr(ydata, 'tolist') else list(ydata),
                    'label': line.get_label()
                })
            for patch in ax.patches:
                axis_data['patches'].append({'x': patch.get_x(), 'width': patch.get_width(), 'height': patch.get_height()})
            for coll in ax.collections:
                offsets = coll.get_offsets()
                if len(offsets) > 0:
                    axis_data['collections'].append({
                        'x': offsets[:, 0].tolist() if hasattr(offsets[:, 0], 'tolist') else list(offsets[:, 0]),
                        'y': offsets[:, 1].tolist() if hasattr(offsets[:, 1], 'tolist') else list(offsets[:, 1])
                    })
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
        self.executor = executor

    def get_figure_hashes(self) -> Dict[str, str]:
        """Get hash fingerprints of all current matplotlib figures."""
        try:
            result = self.executor.run_code(FIGURE_HASH_CODE)
            if result.logs and result.logs.stdout:
                output = "\n".join(result.logs.stdout).strip()
                return json.loads(output)
        except:
            pass
        return {}

    def get_new_figures(self, pre_hashes: Dict[str, str]) -> Set[str]:
        """Get figure numbers that are new or modified since pre_hashes."""
        post_hashes = self.get_figure_hashes()
        new_or_modified = set()

        for fig_num, post_hash in post_hashes.items():
            pre_hash = pre_hashes.get(fig_num)
            if pre_hash is None or pre_hash != post_hash:
                new_or_modified.add(fig_num)

        return new_or_modified

    def close_figures(self, fig_nums: Set[str]) -> None:
        """Close specified figures."""
        if not fig_nums:
            return
        try:
            code = f"import matplotlib.pyplot as plt\nfor n in [{','.join(fig_nums)}]:\n    try: plt.close(n)\n    except: pass"
            self.executor.run_code(code)
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
        charts = [process_axis_group_to_chart(group) for group in axis_groups]
        return [c for c in charts if c]

    def _fetch_raw_axes(self) -> List[Dict[str, Any]]:
        """Execute extraction code in E2B and return raw axis data."""
        try:
            result = self.executor.run_code(RAW_DATA_EXTRACTION)
            if not result.logs or not result.logs.stdout:
                return []

            output = "\n".join(result.logs.stdout).strip()
            data = json.loads(output)

            if isinstance(data, dict) and 'error' in data:
                return []

            return data if isinstance(data, list) else []
        except:
            return []
