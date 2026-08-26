from __future__ import annotations

import base64
import html
import io
import json
import os
from pathlib import Path

from PIL import Image


PROJECT = Path(
    os.environ.get("DBR_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
ROOT = Path(
    os.environ.get(
        "DBR_FIGURE_OUTPUT",
        PROJECT / "report-figures" / "generated",
    )
).expanduser().resolve()
OUTPUT = ROOT / "paper-figure-gallery.html"

manifest = json.loads((ROOT / "figure-manifest.json").read_text(encoding="utf-8"))
main_count = int(manifest["main_figures"])
FIGURES = []
for index, record in enumerate(manifest["figures"], 1):
    if index <= main_count:
        key = f"f{index:02d}"
        label = f"Figure {index}"
    else:
        supplement_index = index - main_count
        key = f"s{supplement_index:02d}"
        label = f"Supplement S{supplement_index}"
    FIGURES.append((key, label, record["title"], Path(record["svg"]).name))


def thumbnail_data_url(svg_path: Path) -> str:
    image_path = ROOT / "png" / f"{svg_path.stem}.png"
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((640, 440), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=42, optimize=True, progressive=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


records = {
    key: {
        "label": label,
        "title": title,
        "src": thumbnail_data_url(ROOT / filename),
    }
    for key, label, title, filename in FIGURES
}

buttons = "\n".join(
    f'<button type="button" data-figure="{key}" aria-pressed="{str(index == 0).lower()}">'
    f"{html.escape(label)}</button>"
    for index, (key, label, _title, _filename) in enumerate(FIGURES)
)

fragment = f"""<div id="paper-figure-gallery" class="paper-figure-gallery">
  <style>
    #paper-figure-gallery {{
      color: var(--foreground);
      font-family: var(--font-sans);
      width: 100%;
    }}
    #paper-figure-gallery .gallery-heading {{
      font-size: 18px;
      font-weight: 650;
      line-height: 1.25;
      margin: 0 0 8px;
    }}
    #paper-figure-gallery .gallery-controls {{
      align-items: center;
      display: flex;
      flex-wrap: wrap;
      gap: 4px 12px;
      margin: 0 0 10px;
    }}
    #paper-figure-gallery button {{
      appearance: none;
      background: transparent;
      border: 0;
      border-bottom: 2px solid transparent;
      color: var(--muted-foreground);
      cursor: pointer;
      font: inherit;
      font-size: 12px;
      line-height: 1.8;
      padding: 0 1px;
    }}
    #paper-figure-gallery button[aria-pressed="true"] {{
      border-bottom-color: var(--viz-series-1);
      color: var(--foreground);
      font-weight: 650;
    }}
    #paper-figure-gallery button:focus-visible {{
      outline: 2px solid var(--ring);
      outline-offset: 2px;
    }}
    #paper-figure-gallery .gallery-title {{
      color: var(--muted-foreground);
      font-size: 12px;
      margin: 0 0 7px;
    }}
    #paper-figure-gallery .gallery-stage {{
      border: 1px solid var(--border);
      overflow: hidden;
      width: 100%;
    }}
    #paper-figure-gallery img {{
      background: white;
      display: block;
      height: auto;
      max-height: 680px;
      object-fit: contain;
      width: 100%;
    }}
    @media (max-width: 440px) {{
      #paper-figure-gallery .gallery-controls {{
        gap: 2px 9px;
      }}
      #paper-figure-gallery button {{
        font-size: 11px;
      }}
    }}
  </style>
  <h2 class="gallery-heading">Publication figure suite</h2>
  <div class="gallery-controls" role="group" aria-label="Select a publication figure">
    {buttons}
  </div>
  <div class="gallery-title" aria-live="polite"></div>
  <div class="gallery-stage">
    <img alt="" />
  </div>
  <script>
    (() => {{
      const root = document.getElementById('paper-figure-gallery');
      const records = {json.dumps(records, ensure_ascii=True, separators=(",", ":"))};
      const image = root.querySelector('img');
      const title = root.querySelector('.gallery-title');
      const buttons = [...root.querySelectorAll('button[data-figure]')];
      function selectFigure(key) {{
        const record = records[key];
        if (!record) return;
        image.src = record.src;
        image.alt = `${{record.label}}. ${{record.title}}`;
        title.textContent = `${{record.label}} · ${{record.title}}`;
        buttons.forEach((button) => {{
          button.setAttribute('aria-pressed', String(button.dataset.figure === key));
        }});
      }}
      buttons.forEach((button) => {{
        button.addEventListener('click', () => selectFigure(button.dataset.figure));
      }});
      selectFigure('f01');
    }})();
  </script>
</div>
"""

OUTPUT.write_text(fragment, encoding="utf-8")
print(OUTPUT)
print(OUTPUT.stat().st_size)
