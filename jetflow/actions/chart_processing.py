"""Shared chart processing utilities for matplotlib chart extraction."""

from typing import List, Dict, Any, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from jetflow.models.chart import Chart, ChartSeries


def group_axes_by_twins(raw_axes: List[Dict]) -> List[List[Dict]]:
    """Group axes that share x-axis into single chart groups."""
    fig_axes = {}
    for ax in raw_axes:
        fig_num = ax['fig_num']
        if fig_num not in fig_axes:
            fig_axes[fig_num] = []
        fig_axes[fig_num].append(ax)

    all_groups = []
    for fig_num, axes in fig_axes.items():
        processed = set()
        for ax in axes:
            if ax['ax_id'] in processed:
                continue
            group = [ax]
            for other_ax in axes:
                if (other_ax['ax_id'] != ax['ax_id'] and
                    other_ax['ax_id'] not in processed and
                    other_ax['ax_id'] in ax['shared_x_ids']):
                    group.append(other_ax)
                    processed.add(other_ax['ax_id'])
            processed.add(ax['ax_id'])
            all_groups.append(group)
    return all_groups


def process_axis_group_to_chart(axis_group: List[Dict]) -> 'Chart':
    """Process a group of twin axes into a single Chart model."""
    from jetflow.models.chart import Chart

    axis_indices = _assign_axis_indices(axis_group)
    metadata = _extract_chart_metadata(axis_group)

    all_series = []
    for ax in axis_group:
        axis_idx = axis_indices[ax['ax_id']]
        series = _extract_series_from_axis(ax, axis_idx, len(all_series))
        all_series.extend(series)

    if not all_series:
        return None

    chart_type = _infer_chart_type(all_series)
    has_multi_axis = any(s.axis == 1 for s in all_series)
    y_label = None if has_multi_axis else metadata.get('ylabel')

    return Chart(
        chart_id=metadata['chart_id'],
        type=chart_type,
        title=metadata.get('title'),
        x_label=metadata.get('xlabel'),
        y_label=y_label,
        series=all_series,
        x_scale=metadata.get('xscale', 'linear'),
        y_scale=metadata.get('yscale', 'linear'),
    )


def _assign_axis_indices(axis_group: List[Dict]) -> Dict[int, int]:
    """Assign axis indices (0=primary, 1=secondary) to twin axes."""
    if len(axis_group) == 1:
        return {axis_group[0]['ax_id']: 0}
    return {ax['ax_id']: idx for idx, ax in enumerate(axis_group)}


def _extract_chart_metadata(axis_group: List[Dict]) -> Dict[str, Any]:
    """Extract chart-level metadata from axis group."""
    first_ax = axis_group[0]
    title, xlabel, ylabel = None, None, None

    for ax in axis_group:
        if ax.get('title') and not title:
            title = ax['title']
        if ax.get('xlabel') and not xlabel:
            xlabel = ax['xlabel']
        if ax.get('ylabel') and not ylabel:
            ylabel = ax['ylabel']

    # Determine chart_id with priority: saved_filename > var_name > fig_label > auto-generated
    chart_id = None
    if first_ax.get('saved_filename'):
        chart_id = first_ax['saved_filename']
    elif first_ax.get('var_name'):
        chart_id = first_ax['var_name']
    elif first_ax.get('fig_label'):
        chart_id = first_ax['fig_label']
    else:
        chart_id = f"fig-{first_ax['fig_num']}-ax-{first_ax['ax_idx']}"

    return {
        'chart_id': chart_id,
        'title': title, 'xlabel': xlabel, 'ylabel': ylabel,
        'xscale': first_ax.get('xscale', 'linear'),
        'yscale': first_ax.get('yscale', 'linear'),
    }


def _extract_series_from_axis(ax: Dict, axis_idx: int, series_count: int) -> List['ChartSeries']:
    """Extract all series from a single axis."""
    from jetflow.models.chart import ChartSeries

    series = []

    for line_data in ax.get('lines', []):
        label = _generate_label(line_data.get('label'), ax.get('ylabel'), len(ax.get('lines', [])), series_count + len(series))
        series.append(ChartSeries(type='line', label=label, x=line_data['x'], y=line_data['y'], axis=axis_idx))

    if ax.get('patches'):
        series.extend(_extract_bar_series(ax, axis_idx, series_count + len(series)))

    for coll_data in ax.get('collections', []):
        label = _generate_label(None, ax.get('ylabel'), len(ax.get('collections', [])), series_count + len(series))
        series.append(ChartSeries(type='scatter', label=label, x=coll_data['x'], y=coll_data['y'], axis=axis_idx))

    return series


def _extract_bar_series(ax: Dict, axis_idx: int, series_count: int) -> List['ChartSeries']:
    """Extract bar series from patches."""
    from jetflow.models.chart import ChartSeries

    patches = ax['patches']
    if not patches:
        return []

    bar_groups = {}
    for patch in patches:
        x_center = round(patch['x'] + patch['width'] / 2, 6)
        if x_center not in bar_groups:
            bar_groups[x_center] = []
        bar_groups[x_center].append(patch['height'])

    x_positions = sorted(bar_groups.keys())
    max_bars_per_pos = max(len(bar_groups[x]) for x in x_positions)
    xtick_labels = ax.get('xtick_labels', [])
    x_values = xtick_labels[:len(x_positions)] if xtick_labels else x_positions

    series = []
    if max_bars_per_pos > 1:
        for bar_idx in range(max_bars_per_pos):
            y_data = [bar_groups[x][bar_idx] if bar_idx < len(bar_groups[x]) else None for x in x_positions]
            series.append(ChartSeries(type='bar', label=f'series-{series_count + len(series) + 1}', x=x_values, y=y_data, axis=axis_idx))
    else:
        y_data = [bar_groups[x][0] for x in x_positions]
        series.append(ChartSeries(type='bar', label=ax.get('ylabel') or f'series-{series_count + 1}', x=x_values, y=y_data, axis=axis_idx))

    return series


def _generate_label(artist_label: str, ylabel: str, num_artists: int, series_count: int) -> str:
    """Generate meaningful label with fallback logic."""
    if artist_label and not artist_label.startswith('_'):
        return artist_label
    if num_artists == 1 and ylabel:
        return ylabel
    return f'series-{series_count + 1}'


def _infer_chart_type(series: List['ChartSeries']) -> str:
    """Infer overall chart type from series types."""
    types = [s.type for s in series]
    if all(t == 'bar' for t in types):
        return 'bar'
    if all(t == 'line' for t in types):
        return 'line'
    if all(t == 'scatter' for t in types):
        return 'scatter'
    return 'mixed'
