"""Chart processing for matplotlib chart extraction."""

from __future__ import annotations

from typing import List, Dict, Any, Optional

from jetflow.models.chart import Chart, ChartSeries, SeriesStyle, YAxis, PieSlice, Axis, ReferenceLine
from jetflow.actions.chart_utils import (
    generate_series_id,
    parse_patches,
    map_x_to_labels,
    is_spurious_collection,
    is_horizontal_bar,
    group_by_x_center,
    group_by_y_center,
)


class ChartProcessor:
    """Processes a group of twin axes into a single Chart model."""

    def __init__(self, axis_group: List[Dict]):
        self._axis_group = axis_group
        self._axis_indices: Dict[int, int] = {}
        self._series_count = 0
        self._metadata: Dict[str, Any] = {}
        self._y_axes: List[YAxis] = []

    def process(self) -> Optional[Chart]:
        """Transform axis group into Chart model."""
        pie_data = self._extract_pie_data()
        if pie_data:
            return self._build_pie_chart(pie_data)
        return self._build_standard_chart()

    def _build_pie_chart(self, pie_data: List[PieSlice]) -> Chart:
        """Build a pie chart from extracted slices."""
        self._extract_metadata()
        return Chart(
            chart_id=self._metadata['chart_id'],
            type='pie',
            title=self._metadata.get('title'),
            subtitle=self._metadata.get('subtitle'),
            pie_data=pie_data,
            citations=self._metadata.get('citations', []),
            data_source=self._metadata.get('data_source'),
        )

    def _build_standard_chart(self) -> Optional[Chart]:
        """Build a standard (non-pie) chart."""
        self._assign_axis_indices()
        self._build_y_axes()
        self._extract_metadata()

        all_series = []
        all_ref_lines = []

        for ax in self._axis_group:
            self._current_ax = ax
            self._current_axis_idx = self._axis_indices[ax['ax_id']]
            series, ref_lines = self._extract_series_from_axis()
            all_series.extend(series)
            all_ref_lines.extend(ref_lines)

        if not all_series and not all_ref_lines:
            return None

        return Chart(
            chart_id=self._metadata['chart_id'],
            type=self._infer_chart_type(all_series),
            orientation=self._detect_orientation(),
            title=self._metadata.get('title'),
            subtitle=self._metadata.get('subtitle'),
            x_axis=Axis(label=self._metadata.get('xlabel'), scale=self._metadata.get('xscale', 'linear')),
            y_axes=self._y_axes,
            series=all_series,
            reference_lines=all_ref_lines,
            citations=self._metadata.get('citations', []),
            data_source=self._metadata.get('data_source'),
        )

    def _assign_axis_indices(self) -> None:
        """Assign y-axis index to each axis in the group."""
        if len(self._axis_group) == 1:
            self._axis_indices = {self._axis_group[0]['ax_id']: 0}
        else:
            self._axis_indices = {ax['ax_id']: idx for idx, ax in enumerate(self._axis_group)}

    def _build_y_axes(self) -> None:
        """Build y_axes list with labels for each axis."""
        self._y_axes = []
        for ax in self._axis_group:
            idx = self._axis_indices[ax['ax_id']]
            position = 'left' if idx == 0 else 'right'
            self._y_axes.append(YAxis(
                label=ax.get('ylabel'),
                scale=ax.get('yscale', 'linear'),
                position=position,
            ))

    def _extract_metadata(self) -> None:
        """Extract chart-level metadata from axis group."""
        first_ax = self._axis_group[0]
        title, xlabel = None, None

        for ax in self._axis_group:
            title = title or ax.get('title')
            xlabel = xlabel or ax.get('xlabel')

        chart_id = (
            first_ax.get('saved_filename') or
            first_ax.get('var_name') or
            first_ax.get('fig_label') or
            f"fig-{first_ax['fig_num']}-ax-{first_ax['ax_idx']}"
        )

        self._metadata = {
            'chart_id': chart_id,
            'title': title,
            'subtitle': first_ax.get('subtitle'),
            'xlabel': xlabel,
            'xscale': first_ax.get('xscale', 'linear'),
            'data_source': first_ax.get('data_source'),
            'citations': first_ax.get('citations', []),
        }

    def _detect_orientation(self) -> str:
        """Detect if chart uses horizontal bars."""
        for ax in self._axis_group:
            patches = ax.get('patches', [])
            if patches and is_horizontal_bar(parse_patches(patches)):
                return 'horizontal'
        return 'vertical'

    def _extract_pie_data(self) -> Optional[List[PieSlice]]:
        """Extract pie slices if this is a pie chart."""
        for ax in self._axis_group:
            raw_pie = ax.get('pie_slices')
            if raw_pie:
                return [
                    PieSlice(
                        label=s['label'],
                        value=s['value'],
                        color=s.get('color'),
                        explode=s.get('explode', 0.0),
                    )
                    for s in raw_pie
                ]
        return None

    def _extract_series_from_axis(self) -> tuple:
        """Extract all series and reference lines from current axis."""
        ax = self._current_ax
        series = []
        reference_lines = []

        xtick_labels = ax.get('xtick_labels', [])
        self._xtick_labels = [l for l in xtick_labels if l.strip()] if xtick_labels else []

        for area_data in (ax.get('area_fills') or []):
            label = area_data.get('label')
            if label and label.startswith('_'):
                label = None
            series.append(ChartSeries(
                id=generate_series_id(),
                type='area',
                label=label or f'area-{self._series_count + len(series) + 1}',
                x=area_data['x'],
                y=area_data['y'],
                y_axis=self._current_axis_idx,
                style=SeriesStyle(color=area_data.get('color'), alpha=area_data.get('alpha')),
            ))

        for line_data in ax.get('lines', []):
            ref_line = self._extract_reference_line(line_data)
            if ref_line:
                reference_lines.append(ref_line)
            else:
                label = self._generate_label(
                    line_data.get('label'),
                    ax.get('ylabel'),
                    len(ax.get('lines', [])),
                    self._series_count + len(series)
                )
                x_values = map_x_to_labels(line_data['x'], self._xtick_labels)
                series.append(ChartSeries(
                    id=generate_series_id(),
                    type='line',
                    label=label,
                    x=x_values,
                    y=line_data['y'],
                    y_axis=self._current_axis_idx,
                    style=SeriesStyle(
                        color=line_data.get('color'),
                        line_style=line_data.get('linestyle'),
                        line_width=line_data.get('linewidth'),
                        marker=line_data.get('marker'),
                        marker_size=line_data.get('markersize'),
                        alpha=line_data.get('alpha'),
                    ),
                ))

        if ax.get('patches'):
            series.extend(self._extract_bar_series())

        valid_collections = [c for c in ax.get('collections', []) if not is_spurious_collection(c)]
        for coll_data in valid_collections:
            label = self._generate_label(None, ax.get('ylabel'), len(valid_collections), self._series_count + len(series))
            x_values = map_x_to_labels(coll_data['x'], self._xtick_labels)
            series.append(ChartSeries(
                id=generate_series_id(),
                type='scatter',
                label=label,
                x=x_values,
                y=coll_data['y'],
                y_axis=self._current_axis_idx,
                style=SeriesStyle(color=coll_data.get('color'), alpha=coll_data.get('alpha')),
            ))

        self._series_count += len(series)
        return series, reference_lines

    def _extract_reference_line(self, line_data: Dict) -> Optional[ReferenceLine]:
        """Extract reference line if this is an axhline/axvline."""
        x, y = line_data.get('x', []), line_data.get('y', [])
        if len(x) < 2 or len(y) < 2:
            return None

        label = line_data.get('label')
        if label and label.startswith('_'):
            label = None

        style = SeriesStyle(
            color=line_data.get('color'),
            line_style=line_data.get('linestyle'),
            line_width=line_data.get('linewidth'),
            alpha=line_data.get('alpha'),
        )

        if len(set(y)) == 1 and len(x) == 2:
            return ReferenceLine(orientation='horizontal', value=y[0], label=label, style=style)

        if len(set(x)) == 1 and len(y) == 2:
            return ReferenceLine(orientation='vertical', value=x[0], label=label, style=style)

        return None

    def _extract_bar_series(self) -> List[ChartSeries]:
        """Extract bar series from patches."""
        ax = self._current_ax
        patches = ax['patches']
        if not patches:
            return []

        self._bar_labels = ax.get('bar_labels', [])
        self._parsed_patches = parse_patches(patches)

        if is_horizontal_bar(self._parsed_patches):
            return self._build_horizontal_bar_series()

        return self._build_vertical_bar_series()

    def _build_horizontal_bar_series(self) -> List[ChartSeries]:
        """Build series for horizontal bar charts."""
        ax = self._current_ax
        y_groups = group_by_y_center(self._parsed_patches)
        y_positions = sorted(y_groups.keys())

        ytick_labels = ax.get('ytick_labels', [])
        non_empty_labels = [l for l in ytick_labels if l.strip()] if ytick_labels else []
        y_values = non_empty_labels[:len(y_positions)] if non_empty_labels else y_positions

        bars_at_first_pos = y_groups[y_positions[0]]
        color = bars_at_first_pos[0].get('color') if bars_at_first_pos else None
        alpha = bars_at_first_pos[0].get('alpha') if bars_at_first_pos else None

        x_data = [y_groups[y][0]['width'] for y in y_positions]
        label = self._bar_labels[0] if self._bar_labels else (ax.get('xlabel') or f'series-{self._series_count + 1}')

        return [ChartSeries(
            id=generate_series_id(),
            type='bar',
            label=label,
            x=list(y_values),
            y=x_data,
            y_axis=self._current_axis_idx,
            style=SeriesStyle(color=color, alpha=alpha),
        )]

    def _build_vertical_bar_series(self) -> List[ChartSeries]:
        """Build series for vertical bar charts."""
        self._x_groups = group_by_x_center(self._parsed_patches)
        self._x_positions = sorted(self._x_groups.keys())
        num_categories = len(self._xtick_labels) or len(self._x_positions)

        bar_type = self._detect_bar_type(num_categories)

        if bar_type == 'stacked':
            return self._build_stacked_series()
        elif bar_type == 'grouped':
            return self._build_grouped_series()
        else:
            return self._build_simple_series()

    def _detect_bar_type(self, num_categories: int) -> str:
        """Detect bar chart type: simple, stacked, or grouped."""
        num_bars_per_pos = max(len(self._x_groups[x]) for x in self._x_positions)

        if num_bars_per_pos > 1:
            for x in self._x_positions:
                bars = self._x_groups[x]
                if len(bars) > 1:
                    bottoms = {round(b['bottom'], 6) for b in bars}
                    if len(bottoms) > 1:
                        return 'stacked'

        if len(self._bar_labels) > 1 and len(self._x_positions) > num_categories:
            bars_per_series = len(self._parsed_patches) // len(self._bar_labels)
            if bars_per_series * len(self._bar_labels) == len(self._parsed_patches):
                return 'grouped'

        return 'simple'

    def _build_stacked_series(self) -> List[ChartSeries]:
        """Build series for stacked bar charts."""
        for x in self._x_positions:
            self._x_groups[x] = sorted(self._x_groups[x], key=lambda b: b['bottom'])

        x_values = self._xtick_labels[:len(self._x_positions)] if self._xtick_labels else self._x_positions
        num_layers = max(len(self._x_groups[x]) for x in self._x_positions)

        series = []
        for layer_idx in range(num_layers):
            y_data = []
            color = None
            alpha = None
            for x in self._x_positions:
                if layer_idx < len(self._x_groups[x]):
                    bar = self._x_groups[x][layer_idx]
                    y_data.append(bar['height'])
                    if color is None:
                        color = bar.get('color')
                        alpha = bar.get('alpha')
                else:
                    y_data.append(None)

            label = self._bar_labels[layer_idx] if layer_idx < len(self._bar_labels) else f'series-{self._series_count + len(series) + 1}'
            series.append(ChartSeries(
                id=generate_series_id(),
                type='bar',
                label=label,
                x=list(x_values),
                y=y_data,
                y_axis=self._current_axis_idx,
                stack_group='default',
                style=SeriesStyle(color=color, alpha=alpha),
            ))
        return series

    def _build_grouped_series(self) -> List[ChartSeries]:
        """Build series for grouped bar charts."""
        bars_per_series = len(self._parsed_patches) // len(self._bar_labels)

        if len(self._xtick_labels) >= bars_per_series:
            x_values = self._xtick_labels[:bars_per_series]
        else:
            x_values = list(range(bars_per_series))

        series = []
        for idx, label in enumerate(self._bar_labels):
            start, end = idx * bars_per_series, (idx + 1) * bars_per_series
            series_patches = sorted(self._parsed_patches[start:end], key=lambda p: p['x_center'])
            y_data = [p['height'] for p in series_patches]
            color = series_patches[0].get('color') if series_patches else None
            alpha = series_patches[0].get('alpha') if series_patches else None

            series.append(ChartSeries(
                id=generate_series_id(),
                type='bar',
                label=label,
                x=list(x_values),
                y=y_data,
                y_axis=self._current_axis_idx,
                style=SeriesStyle(color=color, alpha=alpha),
            ))
        return series

    def _build_simple_series(self) -> List[ChartSeries]:
        """Build series for simple bar charts."""
        ax = self._current_ax
        x_values = self._xtick_labels[:len(self._x_positions)] if self._xtick_labels else self._x_positions
        y_data = [self._x_groups[x][0]['height'] for x in self._x_positions]
        label = self._bar_labels[0] if self._bar_labels else (ax.get('ylabel') or f'series-{self._series_count + 1}')

        first_bar = self._x_groups[self._x_positions[0]][0] if self._x_positions else {}

        return [ChartSeries(
            id=generate_series_id(),
            type='bar',
            label=label,
            x=list(x_values),
            y=y_data,
            y_axis=self._current_axis_idx,
            style=SeriesStyle(color=first_bar.get('color'), alpha=first_bar.get('alpha')),
        )]

    def _generate_label(self, artist_label: str, ylabel: str, num_artists: int, series_idx: int) -> str:
        """Generate a label for a series."""
        if artist_label and not artist_label.startswith('_'):
            return artist_label
        if num_artists == 1 and ylabel:
            return ylabel
        return f'series-{series_idx + 1}'

    def _infer_chart_type(self, series: List[ChartSeries]) -> str:
        """Infer overall chart type from series."""
        types = {s.type for s in series}
        if len(types) == 1:
            return types.pop()
        return 'mixed'
