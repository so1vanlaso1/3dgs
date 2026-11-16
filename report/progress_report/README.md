# 3D Gaussian Splatting research progress report

The LaTeX source follows [`samplereport.tex`](../samplereport.tex) and uses the
university logo from the same template folder. The report lists Nguyen Vu Long
as the sole project member. It uses the user-confirmed school title and
consolidates saved progress through **9 November 2026**, including the
[A05-Abs results](../../docs/A05_ABSGRAD_RESULTS_20261107.md) and
[surface/mesh catch-up](../../docs/SCHOOL_COMPLETION_RESULTS_20261109.md).
The [6 November progress record](../../docs/RESEARCH_PROGRESS_REPORT_20261106.md)
remains historical evidence for earlier work. The current report distinguishes
raw mesh extraction from cleanup/editing validation and keeps supervisor
presentation/feedback pending. No experiments were rerun for this revision.

## Build from the repository

From the repository root, run the following in PowerShell with pdfLaTeX on
your `PATH`. With Ubuntu WSL, replace `pdflatex` in both lines with
`wsl -d Ubuntu -- pdflatex`.

```powershell
Set-Location report/progress_report
New-Item -ItemType Directory -Force build | Out-Null
pdflatex -halt-on-error -file-line-error -interaction=nonstopmode -output-directory=build research_progress.tex
pdflatex -halt-on-error -file-line-error -interaction=nonstopmode -output-directory=build research_progress.tex
```

Repeat the command if LaTeX still requests a rerun (the expanded report can
need a third pass after a clean build). The final log should contain no
undefined references, rerun warnings, or overfull boxes.
The resulting PDF is `build/research_progress.pdf`; copy it to
`research_progress.pdf` after verification to refresh the reader-facing copy.
The report keeps its chart
and bibliography in LaTeX and uses the bundled `figures/hcmut-logo.png`.
On Ubuntu, the TeX Live packages `texlive-latex-recommended`,
`texlive-pictures`, and `lmodern` provide the build dependencies.

## Overleaf-style local editing (VS Code + WSL)

**Installed on 9 November 2026:** Microsoft WSL 1.104.7, LaTeX Workshop
10.20.0 (Windows and Ubuntu remote host), and Ubuntu `latexmk` 4.87.
The report-local `.vscode/settings.json` enables one-second auto-save,
automatic builds on save, SyncTeX, and a PDF tab using `build/`.
Verified the complete editor workflow: a harmless whitespace edit auto-saved,
LaTeX Workshop built the 19-page report successfully with zero logged messages,
and its side-by-side PDF viewer refreshed automatically.

Reopen the configured report from PowerShell:

```powershell
code --new-window --remote wsl+Ubuntu /mnt/d/gsplat/gsplat/report/progress_report
```

Open `research_progress.tex`, then use **LaTeX Workshop: View LaTeX PDF file**
from Ctrl+Shift+P. If VS Code shows **Restricted Mode**, the user must trust
this report folder before extension builds can run. The user enabled trust
for this workspace during setup.

Build-on-save is intentional: Windows-side writes under `/mnt/d` did not
reliably reach WSL file watchers during setup. If the same file is changed
in another editor, reload the current source before editing to avoid save
conflicts. Keep this WSL report window as the editing window.

For a fresh machine, the original setup procedure follows. Before installation
on 9 November, Ubuntu WSL already provided `pdflatex`, but
`latexmk` was not found. The Windows VS Code extension list did not include
WSL or LaTeX Workshop. To set up an editor with automatic builds:

1. Install the Microsoft WSL extension (`ms-vscode-remote.remote-wsl`) in
   Windows VS Code.
2. In an Ubuntu WSL terminal, run `sudo apt update` and then
   `sudo apt install latexmk`.
3. In that terminal, run `cd /mnt/d/gsplat/gsplat/report/progress_report`
   followed by `code .`. Confirm VS Code indicates `WSL: Ubuntu`.
4. Install LaTeX Workshop (`James-Yu.latex-workshop`) in **WSL: Ubuntu**
   from that VS Code window.
5. Open `research_progress.tex`. In Settings, set
   `latex-workshop.latex.autoBuild.run` to `onSave`. Turn on File > Auto Save
   if desired (already enabled in this folder). The recipe uses `latexmk` to
   repeat compilation as needed.
6. Use the command palette command **LaTeX Workshop: View LaTeX PDF file**
   for a PDF preview beside the source. Save edits to rebuild and refresh it.

Builds use the extension's configured output directory; this preview is the
live result. A build under `build/` does not refresh the adjacent distributed
`research_progress.pdf` until it is copied there.

References: [LaTeX Workshop compile settings](https://github.com/James-Yu/LaTeX-Workshop/wiki/Compile)
and [VS Code WSL setup](https://code.visualstudio.com/docs/remote/wsl).
