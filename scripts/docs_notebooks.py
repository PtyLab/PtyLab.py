"""Resolve notebook symlink placeholders produced by Git on Windows."""

from pathlib import Path

from mkdocs.exceptions import PluginError
from mkdocs.plugins import event_priority


@event_priority(-100)
def on_files(files, config):
    # Run after mkdocs-jupyter has created its NotebookFile instances, retaining
    # their documentation URLs while pointing conversion at the real notebooks.
    notebook_root = (Path(config["docs_dir"]).parent / "jupyter_tutorials").resolve()
    for file in files:
        if not file.src_uri.endswith(".ipynb") or not file.abs_src_path:
            continue
        source = Path(file.abs_src_path)
        if source.is_symlink() or source.stat().st_size > 1024:
            continue
        reference = source.read_text(encoding="utf-8").strip()
        if not reference.startswith("../../jupyter_tutorials/"):
            continue
        target = (source.parent / reference).resolve()
        if not target.is_relative_to(notebook_root) or not target.is_file():
            raise PluginError(f"Notebook link {source} has an invalid target: {reference}")
        file.abs_src_path = str(target)
    return files
