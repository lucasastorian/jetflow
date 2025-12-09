"""Chart processing utility functions."""

from __future__ import annotations

from typing import List, Dict
import uuid


def group_axes_by_twins(raw_axes: List[Dict]) -> List[List[Dict]]:
    """Group axes that share x-axis into single chart groups (for twin-axis charts)."""
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


def generate_series_id() -> str:
    """Generate a unique series ID."""
    return str(uuid.uuid4())[:8]


def parse_patches(patches: List[Dict]) -> List[Dict]:
    """Parse raw patch data into normalized format."""
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
            'x_center': round(x + width / 2, 6),
            'color': p.get('color'),
            'alpha': p.get('alpha'),
        })
    return result


def map_x_to_labels(x_values: List, labels: List[str]) -> List:
    """Map numeric x-values to categorical labels if applicable."""
    if not labels or not x_values:
        return x_values

    try:
        indices = [int(round(x)) for x in x_values]
        if not all(abs(x - round(x)) < 0.01 for x in x_values):
            return x_values
        if indices != list(range(len(indices))):
            return x_values
        if not all(0 <= idx < len(labels) for idx in indices):
            return x_values
        return [labels[idx] for idx in indices]
    except (TypeError, ValueError):
        pass

    return x_values


def is_spurious_collection(coll_data: Dict) -> bool:
    """Check if a collection is a spurious artifact (e.g., origin point)."""
    x, y = coll_data.get('x', []), coll_data.get('y', [])
    if len(x) == 1 and len(y) == 1:
        if abs(x[0]) < 0.01 and abs(y[0]) < 0.01:
            return True
    return False


def is_horizontal_bar(parsed: List[Dict]) -> bool:
    """Detect if bars are horizontal (barh) vs vertical (bar)."""
    if not parsed:
        return False

    x_values = [p['x'] for p in parsed]
    heights = [p['height'] for p in parsed]
    widths = [p['width'] for p in parsed]

    x_variance = max(x_values) - min(x_values) if x_values else 0
    height_variance = max(heights) - min(heights) if heights else 0
    width_variance = max(widths) - min(widths) if widths else 0

    if x_variance < 0.01 and height_variance < 0.01 and width_variance > 0.1:
        return True

    return False


def group_by_x_center(patches: List[Dict]) -> Dict[float, List[Dict]]:
    """Group patches by x-center position."""
    groups: Dict[float, List[Dict]] = {}
    for p in patches:
        groups.setdefault(p['x_center'], []).append(p)
    return groups


def group_by_y_center(patches: List[Dict]) -> Dict[float, List[Dict]]:
    """Group patches by y-center position (for horizontal bars)."""
    groups: Dict[float, List[Dict]] = {}
    for p in patches:
        y_center = round(p['bottom'] + p['height'] / 2, 6)
        groups.setdefault(y_center, []).append(p)
    return groups
