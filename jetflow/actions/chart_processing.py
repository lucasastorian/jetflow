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
    xtick_labels = ax.get('xtick_labels', [])
    non_empty_labels = [l for l in xtick_labels if l.strip()] if xtick_labels else []

    # Filter out axhline/axvline (constant horizontal/vertical lines used for reference)
    data_lines = [l for l in ax.get('lines', []) if not _is_reference_line(l)]

    for line_data in data_lines:
        label = _generate_label(line_data.get('label'), ax.get('ylabel'), len(data_lines), series_count + len(series))
        x_values = _map_x_to_labels(line_data['x'], non_empty_labels)
        series.append(ChartSeries(type='line', label=label, x=x_values, y=line_data['y'], axis=axis_idx))

    if ax.get('patches'):
        series.extend(_extract_bar_series(ax, axis_idx, series_count + len(series)))

    # Filter out spurious scatter points (e.g., seaborn legend handles at origin)
    valid_collections = [c for c in ax.get('collections', []) if not _is_spurious_collection(c)]

    for coll_data in valid_collections:
        label = _generate_label(None, ax.get('ylabel'), len(valid_collections), series_count + len(series))
        x_values = _map_x_to_labels(coll_data['x'], non_empty_labels)
        series.append(ChartSeries(type='scatter', label=label, x=x_values, y=coll_data['y'], axis=axis_idx))

    return series


def _extract_bar_series(ax: Dict, axis_idx: int, series_count: int) -> List['ChartSeries']:
    from jetflow.models.chart import ChartSeries

    patches = ax['patches']
    if not patches:
        return []

    bar_labels = ax.get('bar_labels', [])
    parsed = _parse_patches(patches)

    # Detect horizontal vs vertical bars
    is_horizontal = _is_horizontal_bar(parsed)

    if is_horizontal:
        # For horizontal bars: group by y_center, value is width, labels from yticks
        y_groups = _group_by_y_center(parsed)
        y_positions = sorted(y_groups.keys())
        ytick_labels = ax.get('ytick_labels', [])
        non_empty_labels = [l for l in ytick_labels if l.strip()] if ytick_labels else []
        return _build_horizontal_bar_series(y_groups, y_positions, non_empty_labels, bar_labels, ax, series_count, axis_idx, ChartSeries)

    # Vertical bars: original logic
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


def _group_by_y_center(patches: List[Dict]) -> Dict[float, List[Dict]]:
    """Group patches by their y-center position (for horizontal bars)."""
    groups: Dict[float, List[Dict]] = {}
    for p in patches:
        y_center = round(p['bottom'] + p['height'] / 2, 6)
        groups.setdefault(y_center, []).append(p)
    return groups


def _is_horizontal_bar(parsed: List[Dict]) -> bool:
    """Detect if bars are horizontal (barh) vs vertical (bar).

    Horizontal bars: all have x=0 (or same x), width varies (the value), height is constant (bar thickness)
    Vertical bars: x varies (positions), width is constant (bar thickness), height varies (the value)
    """
    if not parsed:
        return False

    # Check if all bars start at x=0 (common for horizontal bars)
    x_values = [p['x'] for p in parsed]
    heights = [p['height'] for p in parsed]
    widths = [p['width'] for p in parsed]

    # If all x values are the same (typically 0) and heights are similar but widths vary -> horizontal
    x_variance = max(x_values) - min(x_values) if x_values else 0
    height_variance = max(heights) - min(heights) if heights else 0
    width_variance = max(widths) - min(widths) if widths else 0

    # Horizontal bars: x is constant (0), height is constant (bar thickness ~0.8), width varies (values)
    # Vertical bars: x varies (positions), width is constant (bar thickness), height varies (values)
    if x_variance < 0.01 and height_variance < 0.01 and width_variance > 0.1:
        return True

    return False


def _build_horizontal_bar_series(y_groups, y_positions, non_empty_labels, bar_labels, ax, series_count, axis_idx, ChartSeries):
    """Build series for horizontal bar charts (barh)."""
    # For simple horizontal bars: one bar per y-position, value is width
    y_values = non_empty_labels[:len(y_positions)] if non_empty_labels else y_positions
    x_data = [y_groups[y][0]['width'] for y in y_positions]  # width is the value

    label = bar_labels[0] if bar_labels else (ax.get('xlabel') or f'series-{series_count + 1}')
    return [ChartSeries(type='bar', label=label, x=list(y_values), y=x_data, axis=axis_idx)]


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

    # Use xtick labels if we have enough of them, otherwise generate indices
    if len(non_empty_labels) >= bars_per_series:
        x_values = non_empty_labels[:bars_per_series]
    else:
        x_values = list(range(bars_per_series))

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


def _map_x_to_labels(x_values: List, labels: List[str]) -> List:
    """Map numeric x-values to categorical labels if applicable.

    Only maps when x-values are sequential 0-based indices (0, 1, 2, 3...)
    which is how seaborn/matplotlib encodes categorical data.
    """
    if not labels or not x_values:
        return x_values

    # Only map if x-values look like 0-based sequential indices
    try:
        indices = [int(round(x)) for x in x_values]
        # Check if they're close to integers
        if not all(abs(x - round(x)) < 0.01 for x in x_values):
            return x_values
        # Check if they start at 0 and are sequential (categorical encoding)
        if indices != list(range(len(indices))):
            return x_values
        # Check indices are in range
        if not all(0 <= idx < len(labels) for idx in indices):
            return x_values
        return [labels[idx] for idx in indices]
    except (TypeError, ValueError):
        pass

    return x_values


def _is_spurious_collection(coll_data: Dict) -> bool:
    """Check if a collection is a spurious artifact (e.g., legend handle at origin)."""
    x, y = coll_data.get('x', []), coll_data.get('y', [])
    # Single point at or near origin is likely a legend handle artifact
    if len(x) == 1 and len(y) == 1:
        if abs(x[0]) < 0.01 and abs(y[0]) < 0.01:
            return True
    return False


def _is_reference_line(line_data: Dict) -> bool:
    """Check if a line is a reference line (axhline/axvline) rather than data."""
    x, y = line_data.get('x', []), line_data.get('y', [])
    if len(x) < 2 or len(y) < 2:
        return False
    # axhline: all y values are the same (horizontal line)
    # axvline: all x values are the same (vertical line)
    # These typically span the full axis range with just 2 points
    if len(set(y)) == 1 and len(x) == 2:
        return True
    if len(set(x)) == 1 and len(y) == 2:
        return True
    return False


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
