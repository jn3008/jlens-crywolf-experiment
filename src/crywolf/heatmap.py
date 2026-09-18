"""Write compact SVG heatmaps from a collected example."""
import gzip
import html
import json
import math
import re
import tempfile
from pathlib import Path


GROUP_PALETTES = {
    "evaluation": ("#f7fbff", "#093A72"),
    "benchmark": ("#f0f5ef", "#1c4f33"),
    "fiction": ("#f1eef3", "#46164F"),
    "simulation": ("#f5f1eb", "#60570B"),
}
GROUP_ORDER = ("evaluation", "benchmark", "fiction", "simulation")


def _color(value):
    value = value.lstrip("#")
    if len(value) != 6 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError(f"Invalid colour {value!r}; use #RRGGBB")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _mix(start, end, amount):
    amount = max(0.0, min(1.0, amount))
    rgb = [round(a + (b - a) * amount) for a, b in zip(start, end)]
    return "#" + "".join(f"{v:02x}" for v in rgb)


def _esc(value):
    return html.escape(str(value), quote=True)


def write_heatmap(run_dir, example, group, output, metric="rank", width=900,
                  cell_height=2, cell_width=None, start_color=None,
                  end_color=None, gamma=1.0, show_tokens=False,
                  token_angle=45.0, token_every=10, token_wrap=15,
                  token_line_height=14.0, layer_labels=5, font_size=11.0,
                  token_style="strip", mark_concept_tokens=False, title=None):
    run = Path(run_dir)
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("Heatmap requires a schema-2 run directory")
    if group not in manifest.get("group_token_ids", {}):
        raise ValueError(f"Unknown group {group!r}; choose from {sorted(manifest['group_token_ids'])}")
    if metric not in {"rank", "logit"}:
        raise ValueError("--metric must be rank or logit")
    if (width < 200 or cell_height <= 0 or token_every < 1 or token_wrap < 0
            or token_line_height <= 0 or layer_labels < 2 or font_size <= 0
            or token_style not in {"strip", "angled"}):
        raise ValueError("width must be >= 200; dimensions, font size, and line height positive; token interval positive; layer labels at least 2")
    default_start, default_end = GROUP_PALETTES.get(group, ("#f7fbff", "#084081"))
    start = _color(start_color or default_start)
    end = _color(end_color or default_end)
    directory = run / "examples" / example
    transcript = json.loads((directory / "transcript.json").read_text())
    if transcript.get("status") != "complete":
        raise ValueError(f"{example}: transcript is not complete")
    layers = manifest["resolved_layers"]
    tokens = transcript["tokens"]
    boundary = transcript["prompt_length"]
    values = {layer: [None] * len(tokens) for layer in layers}
    with gzip.open(directory / "readouts.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            cell = json.loads(line)
            value = cell["groups"][group][metric]
            values[cell["layer"]][cell["position"]] = float(value)
    flat = [v for row in values.values() for v in row if v is not None]
    if len(flat) != len(layers) * len(tokens):
        raise ValueError("Incomplete heatmap coverage")
    if metric == "rank":
        maximum = max(1, int(manifest.get("output_vocab_size", max(flat))))
        def strength(value):
            # Log spacing keeps the long high-rank tail from washing out the map.
            return 1.0 - math.log10(max(1.0, value)) / math.log10(maximum)
        legend_left = f"{math.log10(maximum):.2f}"
        legend_right = "0.00"
    else:
        low, high = min(flat), max(flat)
        def strength(value):
            return 0.5 if high == low else (value - low) / (high - low)
        legend_left, legend_right = f"{low:.3f}", f"{high:.3f}"
    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError("--gamma must be positive")
    margin_left, margin_right = 42, 16
    diagonal_step = token_line_height * abs(math.sin(math.radians(token_angle)))
    header_height = 70 + ((token_wrap * diagonal_step) + 30
                          if show_tokens and token_style == "angled" else 0)
    strip_height = 0
    available = width - margin_left - margin_right
    cw = float(cell_width) if cell_width else available / max(1, len(tokens))
    heatmap_width = cw * len(tokens)
    heat_top = header_height
    rects = []
    for row_index, layer in enumerate(layers):
        y = heat_top + row_index * cell_height
        for position, value in enumerate(values[layer]):
            amount = max(0.0, min(1.0, strength(value))) ** gamma
            x = margin_left + position * cw
            rects.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{max(cw, 0.2):.3f}" height="{cell_height}" fill="{_mix(start, end, amount)}"/>')
    boundary_x = margin_left + boundary * cw
    labels = []
    for position in range(0, len(tokens), token_every):
        x = margin_left + (position + 0.5) * cw
        labels.append(f'<line x1="{x:.3f}" y1="{heat_top - 8}" x2="{x:.3f}" y2="{heat_top}" stroke="#596979" stroke-width="1"/>')
        labels.append(f'<text x="{x:.3f}" y="{heat_top - 10}" text-anchor="middle" class="axis">{position}</text>')
    token_labels = []
    token_strip = []
    if show_tokens and token_style == "angled":
        for position, token in enumerate(tokens):
            x = margin_left + (position + 0.5) * cw
            # Stagger labels in a short repeating staircase.
            offset = token_wrap - (position % (token_wrap + 1))
            y = heat_top - 18 - offset * diagonal_step
            text = token["text"].replace("\n", "\\n")
            # The column marks the beginning of the rotated token.
            token_labels.append(f'<line x1="{x:.3f}" y1="{y + 2:.3f}" x2="{x:.3f}" y2="{heat_top - 8}" stroke="#9aa8b5" stroke-width=".6"/>')
            token_labels.append(f'<text x="{x:.3f}" y="{y:.3f}" text-anchor="start" transform="rotate(-{token_angle:g} {x:.3f} {y:.3f})" class="token">{_esc(text)}</text>')
    heat_bottom = heat_top + len(layers) * cell_height
    concept_markers = []
    concept_marker_height = 0
    if mark_concept_tokens:
        concept_ids = {int(token_id) for token_id in manifest["group_token_ids"][group]}
        concept_marker_y = heat_bottom
        concept_marker_height = 12
        for position, token in enumerate(tokens):
            if int(token["token_id"]) in concept_ids:
                x = margin_left + position * cw
                concept_markers.append(
                    f'<rect x="{x:.3f}" y="{concept_marker_y:.3f}" width="{max(cw, 0.2):.3f}" height="6" fill="#c83b3b"/>'
                )
    if show_tokens and token_style == "strip":
        strip_top = heat_bottom + 20 + concept_marker_height
        strip_box_height = font_size + 4
        annotation_height = font_size * 0.8 + 3 if metric == "rank" else 0
        strip_line_height = strip_box_height + annotation_height + 8
        strip_max_x = margin_left + available
        strip_layout = []
        cursor_x = margin_left
        strip_row = 0
        for position, token in enumerate(tokens):
            text = token["text"].replace("\n", "\\n")
            box_width = max(font_size + 4, len(text) * font_size * 0.62 + 4)
            if cursor_x > margin_left and cursor_x + box_width > strip_max_x:
                strip_row += 1
                cursor_x = margin_left
            strip_layout.append((position, text, cursor_x, strip_row, box_width))
            cursor_x += box_width
        for position, text, x, row, box_width in strip_layout:
            y = strip_top + row * strip_line_height
            token_strength = max(strength(values[layer][position]) for layer in layers) ** gamma
            fill = _mix(start, end, token_strength)
            rgb = _color(fill)
            text_color = "#ffffff" if (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) < 145 else "#111827"
            token_strip.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{box_width:.3f}" height="{strip_box_height:.2f}" fill="{fill}" stroke="#ffffff" stroke-width=".5"/>')
            text_y = y + strip_box_height / 2
            token_strip.append(f'<text x="{x + box_width / 2:.3f}" y="{text_y:.2f}" text-anchor="middle" dominant-baseline="middle" class="strip-token" style="fill:{text_color}">{_esc(text)}</text>')
            if metric == "rank":
                minimum_rank = min(int(values[layer][position]) for layer in layers)
                if minimum_rank < 5:
                    rank_y = y + strip_box_height + font_size * 0.78 + 1
                    token_strip.append(f'<text x="{x + box_width / 2:.3f}" y="{rank_y:.2f}" text-anchor="middle" class="axis">r{minimum_rank}</text>')
        for position in range(0, len(tokens), token_every):
            match = next(item for item in strip_layout if item[0] == position)
            _, _, x, row, box_width = match
            y = strip_top + row * strip_line_height
            token_strip.append(f'<text x="{x + box_width / 2:.3f}" y="{y - 2:.3f}" text-anchor="middle" class="axis">{position}</text>')
        strip_height = (strip_row + 1) * strip_line_height + 10
    label_count = min(layer_labels, len(layers))
    selected_label_indices = sorted({round(i * (len(layers) - 1) / (label_count - 1))
                                     for i in range(label_count)})
    layer_label_marks = []
    for row_index in selected_label_indices:
        layer = layers[row_index]
        y = heat_top + row_index * cell_height + max(1, cell_height * 0.8)
        layer_label_marks.append(f'<text x="{margin_left - 7}" y="{y:.3f}" text-anchor="end" class="axis">{layer}</text>')
    default_title = f"prompt: {transcript['id']} · concept: {group}"
    title = title or default_title
    title_svg = (f'<tspan>prompt: </tspan><tspan class="title-mono">{_esc(transcript["id"])}</tspan>'
                 f'<tspan> · concept: </tspan><tspan class="title-mono">{_esc(group)}</tspan>') \
        if title == default_title else _esc(title)
    footer_height = 38 + strip_height
    svg_width = max(width, math.ceil(margin_left + margin_right + heatmap_width))
    svg_height = header_height + len(layers) * cell_height + footer_height
    axis_label_y = heat_top - 26
    legend_width = 160
    legend_x = svg_width - margin_right - legend_width
    legend_y = 14
    legend_title_svg = ('log<tspan baseline-shift="sub" font-size="75%">10</tspan>(lowest token rank) · '
                        f'{_esc(group)}' if metric == "rank" else f'logit · {_esc(group)}')
    boundary_end = heat_bottom
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}" role="img" aria-labelledby="title desc">
<title id="title">{_esc(title)}</title><desc id="desc">J-Lens {metric} heatmap for {len(tokens)} tokens and {len(layers)} layers. Prompt ends at position {boundary}.</desc>
<style>.title{{font:600 {font_size * 1.35:.2f}px Inter,Arial,sans-serif;fill:#172333}}.title-mono{{font-family:monospace;font-weight:400}}.axis{{font:{font_size * .9:.2f}px Inter,Arial,sans-serif;fill:#536477}}.token{{font:{font_size:.2f}px Inter,Arial,sans-serif;fill:#172333}}.strip-token{{font:{font_size:.2f}px monospace;fill:#172333}}.boundary{{font:{font_size * .9:.2f}px Inter,Arial,sans-serif;fill:#b64b00}}.legend{{font:{font_size:.2f}px Inter,Arial,sans-serif;fill:#536477}}</style>
<text x="16" y="16" class="title">{title_svg}</text>
{''.join(labels)}{''.join(token_labels)}
<g shape-rendering="crispEdges">{''.join(rects)}</g>
<line x1="{boundary_x:.3f}" y1="{heat_top}" x2="{boundary_x:.3f}" y2="{boundary_end}" stroke="#d45b00" stroke-width="2"/>
{''.join(concept_markers)}
{''.join(layer_label_marks)}
{''.join(token_strip)}
<text x="{legend_x + legend_width}" y="{legend_y - 5}" text-anchor="end" class="legend">{legend_title_svg}</text>
<rect x="{legend_x}" y="{legend_y}" width="{legend_width}" height="8" fill="url(#legend)"/>
<defs><linearGradient id="legend" x1="0" x2="1"><stop offset="0" stop-color="{_mix(start, end, 0)}"/><stop offset="1" stop-color="{_mix(start, end, 1)}"/></linearGradient></defs>
<text x="{legend_x}" y="{legend_y + 19}" class="legend">{_esc(legend_left)}</text><text x="{legend_x + legend_width}" y="{legend_y + 19}" text-anchor="end" class="legend">{_esc(legend_right)}</text>
<text x="{margin_left + heatmap_width / 2:.3f}" y="{axis_label_y:.3f}" text-anchor="middle" class="axis">token position</text>
<text x="24" y="{heat_top + len(layers) * cell_height / 2:.3f}" text-anchor="middle" transform="rotate(-90 24 {heat_top + len(layers) * cell_height / 2:.3f})" class="axis">layer</text>
</svg>'''
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return svg_width, svg_height, len(tokens), len(layers)


def write_heatmap_stack(run_dir, example, groups, output, **kwargs):
    # Write several group heatmaps as one stacked SVG
    groups = list(groups)
    order = {group: index for index, group in enumerate(GROUP_ORDER)}
    groups.sort(key=lambda group: (order.get(group, len(order)), group))
    if len(groups) < 2:
        raise ValueError("Heatmap stack needs at least two groups")
    if kwargs.get("show_tokens"):
        raise ValueError("Token display is incompatible with multiple concept groups")
    panels = []
    with tempfile.TemporaryDirectory(prefix="crywolf-heatmaps-") as temporary:
        for index, group in enumerate(groups):
            panel_path = Path(temporary) / f"panel-{index}.svg"
            dimensions = write_heatmap(run_dir, example, group, panel_path, **kwargs)
            svg = panel_path.read_text(encoding="utf-8")
            match = re.match(r'<svg[^>]*\bwidth="([0-9.]+)"[^>]*\bheight="([0-9.]+)"[^>]*>(.*)</svg>\s*$',
                             svg, flags=re.DOTALL)
            if not match:
                raise ValueError("Could not read generated heatmap panel")
            panel_content = match.group(3).replace('id="legend"', f'id="legend-{index}"')
            panel_content = panel_content.replace('url(#legend)', f'url(#legend-{index})')
            # A standalone panel reserves a footer; stacked panels do not need it.
            trim = 32 if kwargs.get("mark_concept_tokens") else 38
            panel_height = max(1.0, float(match.group(2)) - trim)
            panels.append((float(match.group(1)), panel_height, panel_content, dimensions))
    gap = 14
    svg_width = max(panel[0] for panel in panels)
    svg_height = sum(panel[1] for panel in panels) + gap * (len(panels) - 1)
    content = []
    y = 0.0
    for index, (panel_width, panel_height, panel_content, _) in enumerate(panels):
        content.append(f'<g transform="translate(0 {y:.3f})">{panel_content}</g>')
        y += panel_height
        if index + 1 < len(panels):
            content.append(f'<line x1="0" y1="{y + gap / 2:.3f}" x2="{svg_width:.3f}" y2="{y + gap / 2:.3f}" stroke="#cbd5e1" stroke-width="1"/>')
            y += gap
    title = f"{example} · stacked concept groups"
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" '
           f'viewBox="0 0 {svg_width} {svg_height}" role="img" aria-labelledby="title desc">'
           f'<title id="title">{_esc(title)}</title>'
           f'<desc id="desc">Stacked J-Lens heatmaps for concept groups: {_esc(", ".join(groups))}.</desc>'
           f'{"".join(content)}</svg>')
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    first = panels[0][3]
    return svg_width, svg_height, first[2], first[3]
