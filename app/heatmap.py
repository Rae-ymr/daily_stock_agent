"""
Renders the /heatmap page: a diverging-color grid of daily % change per
ticker. Diverging (not sequential) because the data has polarity — up vs.
down around a 0% baseline, not just "how much."

Color poles reuse the documented categorical hexes from the design system
(slot 1 blue / slot 8 red, light + dark), per the palette's stated
diverging pair (blue <-> red, neutral gray midpoint).
"""

_POLES = {
    "light": {"neg": "#e34948", "pos": "#2a78d6", "mid": "#f0efec"},
    "dark": {"neg": "#e66767", "pos": "#3987e5", "mid": "#383835"},
}


def _lerp_hex(hex_a: str, hex_b: str, t: float) -> str:
    """Linear RGB interpolation between two hex colors, t in [0, 1]."""
    a = tuple(int(hex_a[i : i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(hex_b[i : i + 2], 16) for i in (1, 3, 5))
    rgb = tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))
    return "#" + "".join(f"{c:02x}" for c in rgb)


def _relative_luminance(hex_color: str) -> float:
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _diverging_color(pct_change: float, max_abs: float, mode: str) -> str:
    poles = _POLES[mode]
    if max_abs <= 0:
        return poles["mid"]
    t = min(abs(pct_change) / max_abs, 1.0)
    pole = poles["pos"] if pct_change >= 0 else poles["neg"]
    return _lerp_hex(poles["mid"], pole, t)


def _text_color_for(fill_hex: str) -> str:
    return "#0b0b0b" if _relative_luminance(fill_hex) > 0.5 else "#ffffff"


def build_heatmap_html(cells: list[dict]) -> str:
    """
    cells: [{"ticker": str, "pct_change": float, "close": float}, ...]
    """
    max_abs = max((abs(c["pct_change"]) for c in cells), default=0) or 1.0

    cell_html = []
    for c in cells:
        light_fill = _diverging_color(c["pct_change"], max_abs, "light")
        dark_fill = _diverging_color(c["pct_change"], max_abs, "dark")
        light_text = _text_color_for(light_fill)
        dark_text = _text_color_for(dark_fill)
        sign = "+" if c["pct_change"] >= 0 else ""
        cell_html.append(
            f'<div class="cell" tabindex="0" '
            f'style="--fill-light:{light_fill}; --fill-dark:{dark_fill}; '
            f'--text-light:{light_text}; --text-dark:{dark_text};" '
            f'data-ticker="{c["ticker"]}" '
            f'data-pct="{sign}{c["pct_change"]:.2f}%" '
            f'data-close="{c["close"]:.2f}">'
            f'<span class="cell-ticker"></span>'
            f'<span class="cell-pct"></span>'
            f"</div>"
        )

    table_rows = "".join(
        f'<tr><td>{c["ticker"]}</td>'
        f'<td>{"+" if c["pct_change"] >= 0 else ""}{c["pct_change"]:.2f}%</td>'
        f'<td>{c["close"]:.2f}</td></tr>'
        for c in cells
    )

    legend_light = (
        f'linear-gradient(to right, {_POLES["light"]["neg"]}, '
        f'{_POLES["light"]["mid"]}, {_POLES["light"]["pos"]})'
    )
    legend_dark = (
        f'linear-gradient(to right, {_POLES["dark"]["neg"]}, '
        f'{_POLES["dark"]["mid"]}, {_POLES["dark"]["pos"]})'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Daily Stock Heatmap</title>
<style>
  .viz-root {{
    color-scheme: light;
    --surface-1: #fcfcfb;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --border: rgba(11,11,11,0.10);
    --legend-gradient: {legend_light};
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--surface-1);
    color: var(--text-primary);
    padding: 24px;
    max-width: 720px;
    margin: 0 auto;
  }}
  @media (prefers-color-scheme: dark) {{
    .viz-root {{
      color-scheme: dark;
      --surface-1: #1a1a19;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --border: rgba(255,255,255,0.10);
      --legend-gradient: {legend_dark};
    }}
  }}
  h1 {{ font-size: 18px; margin: 0 0 2px; }}
  .subtitle {{ font-size: 13px; color: var(--text-secondary); margin: 0 0 20px; }}

  .legend {{ margin-bottom: 20px; }}
  .legend-bar {{
    height: 8px;
    border-radius: 4px;
    background: var(--legend-gradient);
  }}
  .legend-ticks {{
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 4px;
  }}

  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 2px;
    background: var(--surface-1);
  }}
  .cell {{
    background: var(--fill-light);
    color: var(--text-light);
    border-radius: 6px;
    padding: 14px 12px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    cursor: default;
    outline: none;
    transition: filter 0.1s ease;
    position: relative;
  }}
  @media (prefers-color-scheme: dark) {{
    .cell {{ background: var(--fill-dark); color: var(--text-dark); }}
  }}
  .cell:hover, .cell:focus-visible {{
    filter: brightness(1.08);
    box-shadow: 0 0 0 2px var(--border);
  }}
  .cell-ticker {{ font-weight: 600; font-size: 14px; }}
  .cell-pct {{ font-size: 13px; font-variant-numeric: tabular-nums; }}

  .tooltip {{
    position: fixed;
    pointer-events: none;
    background: var(--text-primary);
    color: var(--surface-1);
    font-size: 12px;
    padding: 6px 10px;
    border-radius: 4px;
    opacity: 0;
    transform: translate(-50%, -100%);
    transition: opacity 0.08s ease;
    z-index: 10;
    white-space: nowrap;
  }}
  .tooltip.visible {{ opacity: 1; }}
  .tooltip strong {{ font-variant-numeric: tabular-nums; }}

  .table-toggle {{
    margin-top: 20px;
    font-size: 13px;
    background: none;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 6px 12px;
    color: var(--text-secondary);
    cursor: pointer;
  }}
  table {{
    display: none;
    margin-top: 16px;
    border-collapse: collapse;
    font-size: 13px;
    width: 100%;
  }}
  table.visible {{ display: table; }}
  th, td {{
    text-align: left;
    padding: 6px 10px;
    border-bottom: 1px solid var(--border);
    font-variant-numeric: tabular-nums;
  }}
  th {{ color: var(--text-muted); font-weight: 500; }}
</style>
</head>
<body>
<div class="viz-root">
  <h1>Daily Stock Heatmap</h1>
  <p class="subtitle">% change over the last 5 trading days</p>

  <div class="legend">
    <div class="legend-bar"></div>
    <div class="legend-ticks">
      <span>-{max_abs:.1f}%</span>
      <span>0%</span>
      <span>+{max_abs:.1f}%</span>
    </div>
  </div>

  <div class="grid" id="grid">
    {"".join(cell_html)}
  </div>

  <button class="table-toggle" id="table-toggle">Show as table</button>
  <table id="data-table">
    <thead><tr><th>Ticker</th><th>% Change</th><th>Last Close</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
  const tooltip = document.getElementById('tooltip');
  const cells = document.querySelectorAll('.cell');

  cells.forEach(cell => {{
    cell.querySelector('.cell-ticker').textContent = cell.dataset.ticker;
    cell.querySelector('.cell-pct').textContent = cell.dataset.pct;

    const showTooltip = (e) => {{
      tooltip.innerHTML = '';
      const strong = document.createElement('strong');
      strong.textContent = cell.dataset.pct;
      tooltip.appendChild(strong);
      tooltip.appendChild(document.createTextNode(
        ' — ' + cell.dataset.ticker + ' — last close $' + cell.dataset.close
      ));
      tooltip.classList.add('visible');
      const x = e.clientX !== undefined ? e.clientX : cell.getBoundingClientRect().left + cell.offsetWidth / 2;
      const y = e.clientY !== undefined ? e.clientY : cell.getBoundingClientRect().top;
      tooltip.style.left = x + 'px';
      tooltip.style.top = (y - 12) + 'px';
    }};
    const hideTooltip = () => tooltip.classList.remove('visible');

    cell.addEventListener('pointermove', showTooltip);
    cell.addEventListener('pointerleave', hideTooltip);
    cell.addEventListener('focus', showTooltip);
    cell.addEventListener('blur', hideTooltip);
  }});

  const toggleBtn = document.getElementById('table-toggle');
  const table = document.getElementById('data-table');
  const grid = document.getElementById('grid');
  toggleBtn.addEventListener('click', () => {{
    const showing = table.classList.toggle('visible');
    grid.style.display = showing ? 'none' : 'grid';
    toggleBtn.textContent = showing ? 'Show as heatmap' : 'Show as table';
  }});
</script>
</body>
</html>"""
