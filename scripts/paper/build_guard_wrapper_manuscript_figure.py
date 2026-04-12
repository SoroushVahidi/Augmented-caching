#!/usr/bin/env python3
"""Build manuscript schematic: guarded evict_value_v1 control flow.

Writes vector PDF + PNG under figures/manuscript/ and a LaTeX snippet under
reports/manuscript_artifacts/latex_snippets/. Not part of build_kbs_main_manuscript_artifacts.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from manuscript_figure_common import (  # noqa: E402
    apply_manuscript_matplotlib_style,
    make_guard_wrapper_schematic_figure,
    repo_root,
    save_figure_pdf_png,
)


def main() -> None:
    root = repo_root()
    fig_dir = root / "figures" / "manuscript"
    snippet_dir = root / "reports" / "manuscript_artifacts" / "latex_snippets"
    caption_dir = root / "reports" / "manuscript_artifacts"

    apply_manuscript_matplotlib_style()
    fig = make_guard_wrapper_schematic_figure()
    pdf, png = save_figure_pdf_png(fig, fig_dir, "figure6_guard_wrapper_evict_value_v1")

    cap = (
        r"Optional guard wrapper (\texttt{evict\_value\_v1\_guarded}). "
        r"On full-cache misses the base learned policy scores residents. "
        r"A lightweight \emph{early-return} detector tracks whether the base victim is quickly re-requested; "
        r"suspicious events in a bounded time window can trigger a fixed-duration fallback to \texttt{lru} or \texttt{marker}. "
        r"Both policies are stepped each request so the wrapper can switch modes with consistent shadow state. "
        r"This is a practical empirical control layer; it is not asserted as a competitive robustness theorem."
    )
    snippet = (
        "\\begin{figure}[t]\n\\centering\n"
        "\\includegraphics[width=\\columnwidth]{figures/manuscript/figure6_guard_wrapper_evict_value_v1.pdf}\n"
        f"\\caption{{{cap}}}\n"
        "\\label{fig:guard-wrapper-evict-value-v1}\n"
        "\\end{figure}\n"
    )

    snippet_dir.mkdir(parents=True, exist_ok=True)
    (snippet_dir / "figure6_guard_wrapper_snippet.tex").write_text(snippet, encoding="utf-8")

    plain = (
        "Optional guard wrapper (evict_value_v1_guarded). On full-cache misses the base learned policy scores residents. "
        "A lightweight early-return detector tracks whether the base victim is quickly re-requested; suspicious events in a "
        "bounded time window can trigger a fixed-duration fallback to lru or marker. Both policies are stepped each request "
        "for consistent shadow state (see src/lafc/policies/guard_wrapper.py). This is an empirical control layer described "
        "in-repository; it is not asserted as a competitive robustness theorem."
    )
    (caption_dir / "figure6_guard_wrapper_caption.md").write_text(
        "## Figure: guard wrapper schematic — caption (plain text)\n\n" + plain + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    print(f"Wrote {snippet_dir / 'figure6_guard_wrapper_snippet.tex'}")


if __name__ == "__main__":
    main()
