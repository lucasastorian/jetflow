"""Chart extraction from local matplotlib figures."""

from typing import List, Set, Dict, Any
import matplotlib.pyplot as plt

from jetflow.actions.chart_utils import group_axes_by_twins
from jetflow.actions.chart_processing import ChartProcessor


class LocalChartExtractor:
    """Extracts chart data from local matplotlib figures."""

    def extract(self, fig_nums: Set[str]) -> List['Chart']:
        """Extract charts from specified figure numbers."""
        if not fig_nums:
            return []

        raw_axes = self._get_raw_axes(fig_nums)
        if not raw_axes:
            return []

        axis_groups = group_axes_by_twins(raw_axes)
        return [c for c in (ChartProcessor(g).process() for g in axis_groups) if c]

    def _get_raw_axes(self, fig_nums: Set[str]) -> List[Dict[str, Any]]:
        raw_axes = []
        target_nums = {int(n) for n in fig_nums}
        fig_var_names = self._find_figure_var_names()

        for fig_num in plt.get_fignums():
            if fig_num not in target_nums:
                continue

            fig = plt.figure(fig_num)
            fig_metadata = {
                'fig_label': fig.get_label() or None,
                'saved_filename': getattr(fig, '_jetflow_chart_id', None),
                'var_name': fig_var_names.get(fig_num),
                'subtitle': getattr(fig, 'subtitle', None),
                'data_source': getattr(fig, 'data_source', None),
                'citations': getattr(fig, 'citations', []),
            }

            for ax_idx, ax in enumerate(fig.get_axes()):
                axis_data = self._extract_axis_data(fig_num, ax_idx, ax, fig.get_axes())
                axis_data.update(fig_metadata)
                raw_axes.append(axis_data)

        return raw_axes

    def _find_figure_var_names(self) -> Dict[int, str]:
        try:
            import inspect
            fig_var_names = {}
            frame = inspect.currentframe()
            while frame:
                for var_name, var_value in frame.f_locals.items():
                    if isinstance(var_value, plt.Figure) and not var_name.startswith('_'):
                        fig_var_names[var_value.number] = var_name
                frame = frame.f_back
            return fig_var_names
        except:
            return {}

    def _extract_axis_data(self, fig_num: int, ax_idx: int, ax, all_axes) -> Dict[str, Any]:
        shared_x_ids = [id(other) for other in all_axes if other != ax and ax.get_shared_x_axes().joined(ax, other)]

        return {
            'fig_num': fig_num,
            'ax_idx': ax_idx,
            'ax_id': id(ax),
            'title': ax.get_title() or None,
            'xlabel': ax.get_xlabel() or None,
            'ylabel': ax.get_ylabel() or None,
            'xscale': ax.get_xscale(),
            'yscale': ax.get_yscale(),
            'shared_x_ids': shared_x_ids,
            'lines': self._extract_lines(ax),
            'patches': self._extract_patches(ax),
            'collections': self._extract_collections(ax),
            'bar_labels': self._extract_bar_labels(ax),
            'xtick_labels': [t.get_text() for t in ax.get_xticklabels()],
            'ytick_labels': [t.get_text() for t in ax.get_yticklabels()]
        }

    def _extract_lines(self, ax) -> List[Dict]:
        lines = []
        for line in ax.get_lines():
            xdata, ydata = line.get_xdata(), line.get_ydata()
            lines.append({
                'x': xdata.tolist() if hasattr(xdata, 'tolist') else list(xdata),
                'y': ydata.tolist() if hasattr(ydata, 'tolist') else list(ydata),
                'label': line.get_label()
            })
        return lines

    def _extract_patches(self, ax) -> List[Dict]:
        return [
            {'x': p.get_x(), 'y': p.get_y(), 'width': p.get_width(), 'height': p.get_height()}
            for p in ax.patches
        ]

    def _extract_collections(self, ax) -> List[Dict]:
        collections = []
        for coll in ax.collections:
            offsets = coll.get_offsets()
            if len(offsets) > 0:
                collections.append({
                    'x': offsets[:, 0].tolist() if hasattr(offsets[:, 0], 'tolist') else list(offsets[:, 0]),
                    'y': offsets[:, 1].tolist() if hasattr(offsets[:, 1], 'tolist') else list(offsets[:, 1])
                })
        return collections

    def _extract_bar_labels(self, ax) -> List[str]:
        try:
            from matplotlib.container import BarContainer
            labels = [
                c.get_label() for c in ax.containers
                if isinstance(c, BarContainer) and c.get_label() and not c.get_label().startswith('_')
            ]
            if labels:
                return labels
        except:
            pass

        try:
            from matplotlib.container import BarContainer
            handles, lbls = ax.get_legend_handles_labels()
            return [lbl for h, lbl in zip(handles, lbls) if isinstance(h, BarContainer)]
        except:
            return []
