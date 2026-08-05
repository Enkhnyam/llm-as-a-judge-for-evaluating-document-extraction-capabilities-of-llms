"""Run the checks and write their output to one HTML page, for sharing.

    python checks/report.py            -> artifacts/checks_report.html

Same script list as run.py. Local only; the page is not committed.
"""
import html
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from run import SCRIPTS

HERE = Path(__file__).parent
OUT = HERE.parent / "artifacts" / "checks_report.html"

STYLE = """
body { font: 14px/1.55 -apple-system, system-ui, sans-serif; margin: 0; background: #f6f7f9; color: #1a1a1a; }
header { background: #1f2937; color: #fff; padding: 22px 32px; }
header h1 { margin: 0; font-size: 19px; }
header p { margin: 4px 0 0; opacity: .75; font-size: 13px; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 32px 60px; }
nav { background: #fff; border: 1px solid #e3e6ea; border-radius: 8px; padding: 12px 18px; margin-bottom: 24px; }
nav a { display: inline-block; margin: 3px 14px 3px 0; color: #2563eb; text-decoration: none; font-size: 13px; }
section { background: #fff; border: 1px solid #e3e6ea; border-radius: 8px; margin-bottom: 18px; overflow: hidden; }
h2 { margin: 0; padding: 12px 18px; font-size: 14px; background: #f0f2f5; border-bottom: 1px solid #e3e6ea; font-family: ui-monospace, Menlo, monospace; }
.doc { padding: 12px 18px; color: #4b5563; font-size: 13px; border-bottom: 1px solid #eef0f3; white-space: pre-wrap; }
pre { margin: 0; padding: 16px 18px; font: 12.5px/1.5 ui-monospace, Menlo, monospace; overflow-x: auto; }
.failed h2 { background: #fee2e2; }
"""


def docstring_of(path):
    text = path.read_text()
    if not text.startswith('"""'):
        return ""
    return text[3:text.index('"""', 3)].strip()


sections = []
links = []
for name in SCRIPTS:
    path = HERE / name
    result = subprocess.run([sys.executable, str(path)], cwd=HERE, text=True,
                            capture_output=True,
                            env={**__import__("os").environ, "PYTHONPATH": str(HERE)})
    output = result.stdout.rstrip() or result.stderr.rstrip()
    anchor = name.replace("/", "-").replace(".py", "")
    failed = " failed" if result.returncode else ""
    links.append(f'<a href="#{anchor}">{html.escape(name)}</a>')
    sections.append(
        f'<section class="{failed}" id="{anchor}"><h2>{html.escape(name)}</h2>'
        f'<div class="doc">{html.escape(docstring_of(path))}</div>'
        f'<pre>{html.escape(output)}</pre></section>')
    print(("failed  " if failed else "ok      ") + name)

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(
    f"<!doctype html><meta charset='utf-8'><title>Checks</title><style>{STYLE}</style>"
    f"<header><h1>LLM-as-a-judge &mdash; results</h1>"
    f"<p>Generated {datetime.now():%Y-%m-%d %H:%M} from the data on disk.</p></header>"
    f"<main><nav>{''.join(links)}</nav>{''.join(sections)}</main>")
print(f"\nwrote {OUT}")
