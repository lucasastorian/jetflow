"""Chart extraction from local matplotlib figures."""

from typing import List, Set, Dict, Any
import matplotlib.pyplot as plt

from jetflow.actions.chart_processing import group_axes_by_twins, process_axis_group_to_chart


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
        charts = [process_axis_group_to_chart(group) for group in axis_groups]
        return [c for c in charts if c]

    def _get_raw_axes(self, fig_nums: Set[str]) -> List[Dict[str, Any]]:
        """Extract raw axis data from matplotlib figures."""
        raw_axes = []
        target_nums = {int(n) for n in fig_nums}

        for fig_num in plt.get_fignums():
            if fig_num not in target_nums:
                continue

            fig = plt.figure(fig_num)
            for ax_idx, ax in enumerate(fig.get_axes()):
                raw_axes.append(self._extract_axis_data(fig_num, ax_idx, ax, fig.get_axes()))

        return raw_axes

    def _extract_axis_data(self, fig_num: int, ax_idx: int, ax, all_axes) -> Dict[str, Any]:
        """Extract data from a single axis."""
        shared_x_ids = [id(other) for other in all_axes if other != ax and ax.get_shared_x_axes().joined(ax, other)]

        axis_data = {
            'fig_num': fig_num,
            'ax_idx': ax_idx,
            'ax_id': id(ax),
            'title': ax.get_title() or None,
            'xlabel': ax.get_xlabel() or None,
            'ylabel': ax.get_ylabel() or None,
            'xscale': ax.get_xscale(),
            'yscale': ax.get_yscale(),
            'shared_x_ids': shared_x_ids,
            'lines': [],
            'patches': [],
            'collections': [],
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

        return axis_data
