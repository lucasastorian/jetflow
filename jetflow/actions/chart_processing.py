"""Shared chart processing utilities for matplotlib chart extraction."""

from typing import List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from jetflow.models.chart import Chart, ChartSeries


def group_axes_by_twins(raw_axes: List[Dict]) -> List[List[Dict]]:
    """Group axes that share x-axis into single chart groups."""
    fig_axes: Dict[int, List[Dict]] = {}
    for ax in raw_axes:
        fig_axes.setdefault(ax['fig_num'], []).append(ax)

    all_groups = []
    for axes in fig_axes.values():
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
        all_series.extend(_extract_series_from_axis(ax, axis_idx, len(all_series)))

    if not all_series:
        return None

    has_multi_axis = any(s.axis == 1 for s in all_series)

    return Chart(
        chart_id=metadata['chart_id'],
        type=_infer_chart_type(all_series),
        title=metadata.get('title'),
        subtitle=metadata.get('subtitle'),
        x_label=metadata.get('xlabel'),
        y_label=None if has_multi_axis else metadata.get('ylabel'),
        series=all_series,
        x_scale=metadata.get('xscale', 'linear'),
        y_scale=metadata.get('yscale', 'linear'),
        data_source=metadata.get('data_source'),
        citations=metadata.get('citations', []),
    )


def _assign_axis_indices(axis_group: List[Dict]) -> Dict[int, int]:
    if len(axis_group) == 1:
        return {axis_group[0]['ax_id']: 0}
    return {ax['ax_id']: idx for idx, ax in enumerate(axis_group)}


def _extract_chart_metadata(axis_group: List[Dict]) -> Dict[str, Any]:
    first_ax = axis_group[0]
    title, xlabel, ylabel = None, None, None

    for ax in axis_group:
        title = title or ax.get('title')
        xlabel = xlabel or ax.get('xlabel')
        ylabel = ylabel or ax.get('ylabel')

    chart_id = (
        first_ax.get('saved_filename') or
        first_ax.get('var_name') or
        first_ax.get('fig_label') or
        f"fig-{first_ax['fig_num']}-ax-{first_ax['ax_idx']}"
    )

    return {
        'chart_id': chart_id,
        'title': title,
        'subtitle': first_ax.get('subtitle'),
        'xlabel': xlabel,
        'ylabel': ylabel,
        'xscale': first_ax.get('xscale', 'linear'),
        'yscale': first_ax.get('yscale', 'linear'),
        'data_source': first_ax.get('data_source'),
        'citations': first_ax.get('citations', []),
    }


def _extract_series_from_axis(ax: Dict, axis_idx: int, series_count: int) -> List['ChartSeries']:
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
    from jetflow.models.chart import ChartSeries

    patches = ax['patches']
    if not patches:
        return []

    bar_labels = ax.get('bar_labels', [])
    parsed = _parse_patches(patches)
    x_groups = _group_by_x_center(parsed)
    x_positions = sorted(x_groups.keys())

    xtick_labels = ax.get('xtick_labels', [])
    non_empty_labels = [l for l in xtick_labels if l.strip()] if xtick_labels else []
    num_categories = len(non_empty_labels) or len(x_positions)

    bar_type = _detect_bar_type(x_groups, x_positions, bar_labels, parsed, num_categories)

    if bar_type == 'stacked':
        return _build_stacked_series(x_groups, x_positions, non_empty_labels, bar_labels, series_count, axis_idx, ChartSeries)
    elif bar_type == 'grouped':
        return _build_grouped_series(parsed, bar_labels, non_empty_labels, axis_idx, ChartSeries)
    else:
        return _build_simple_series(x_groups, x_positions, non_empty_labels, bar_labels, ax, series_count, axis_idx, ChartSeries)


def _parse_patches(patches: List[Dict]) -> List[Dict]:
    result = []
    for p in patches:
        x = float(p['x']) if isinstance(p['x'], str) else p['x']
        width = float(p['width']) if isinstance(p['width'], str) else p['width']
        height = float(p['height']) if isinstance(p['height'], str) else p['height']
        bottom = p.get('y', 0)
        bottom = float(bottom) if isinstance(bottom, str) else bottom
        result.append({
            'x': x,
            'width': width,
            'height': height,
            'bottom': bottom,
            'x_center': round(x + width / 2, 6)
        })
    return result


def _group_by_x_center(patches: List[Dict]) -> Dict[float, List[Dict]]:
    groups: Dict[float, List[Dict]] = {}
    for p in patches:
        groups.setdefault(p['x_center'], []).append(p)
    return groups


def _detect_bar_type(x_groups: Dict, x_positions: List, bar_labels: List, parsed: List, num_categories: int) -> str:
    num_bars_per_pos = max(len(x_groups[x]) for x in x_positions)

    if num_bars_per_pos > 1:
        for x in x_positions:
            bars = x_groups[x]
            if len(bars) > 1:
                bottoms = {round(b['bottom'], 6) for b in bars}
                if len(bottoms) > 1:
                    return 'stacked'

    if len(bar_labels) > 1 and len(x_positions) > num_categories:
        bars_per_series = len(parsed) // len(bar_labels)
        if bars_per_series * len(bar_labels) == len(parsed):
            return 'grouped'

    return 'simple'


def _build_stacked_series(x_groups, x_positions, non_empty_labels, bar_labels, series_count, axis_idx, ChartSeries):
    for x in x_positions:
        x_groups[x] = sorted(x_groups[x], key=lambda b: b['bottom'])

    x_values = non_empty_labels[:len(x_positions)] if non_empty_labels else x_positions
    num_layers = max(len(x_groups[x]) for x in x_positions)

    series = []
    for layer_idx in range(num_layers):
        y_data = [
            x_groups[x][layer_idx]['height'] if layer_idx < len(x_groups[x]) else None
            for x in x_positions
        ]
        label = bar_labels[layer_idx] if layer_idx < len(bar_labels) else f'series-{series_count + len(series) + 1}'
        series.append(ChartSeries(type='bar', label=label, x=list(x_values), y=y_data, axis=axis_idx))
    return series


def _build_grouped_series(parsed, bar_labels, non_empty_labels, axis_idx, ChartSeries):
    bars_per_series = len(parsed) // len(bar_labels)
    x_values = non_empty_labels[:bars_per_series] if non_empty_labels else list(range(bars_per_series))

    series = []
    for idx, label in enumerate(bar_labels):
        start, end = idx * bars_per_series, (idx + 1) * bars_per_series
        series_patches = sorted(parsed[start:end], key=lambda p: p['x_center'])
        y_data = [p['height'] for p in series_patches]
        series.append(ChartSeries(type='bar', label=label, x=list(x_values), y=y_data, axis=axis_idx))
    return series


def _build_simple_series(x_groups, x_positions, non_empty_labels, bar_labels, ax, series_count, axis_idx, ChartSeries):
    x_values = non_empty_labels[:len(x_positions)] if non_empty_labels else x_positions
    y_data = [x_groups[x][0]['height'] for x in x_positions]
    label = bar_labels[0] if bar_labels else (ax.get('ylabel') or f'series-{series_count + 1}')
    return [ChartSeries(type='bar', label=label, x=list(x_values), y=y_data, axis=axis_idx)]


def _generate_label(artist_label: str, ylabel: str, num_artists: int, series_count: int) -> str:
    if artist_label and not artist_label.startswith('_'):
        return artist_label
    if num_artists == 1 and ylabel:
        return ylabel
    return f'series-{series_count + 1}'


def _infer_chart_type(series: List['ChartSeries']) -> str:
    types = {s.type for s in series}
    if types == {'bar'}:
        return 'bar'
    if types == {'line'}:
        return 'line'
    if types == {'scatter'}:
        return 'scatter'
    return 'mixed'
