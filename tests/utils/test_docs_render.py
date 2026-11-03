import pytest

docs = pytest.importorskip("PtyLab.utils.docs")

SCRIPT = """# %% [markdown]
# # Title

# %%
x = 1

# %% [markdown]
# Some text.

# %%
print(x)
"""


@pytest.fixture
def example(tmp_path, monkeypatch):
    """A one-page docs folder; `runs` counts how often the script is executed."""
    monkeypatch.setattr(docs, "PAGES", tmp_path)
    script = tmp_path / "example_test.py"
    script.write_text(SCRIPT, encoding="utf-8")
    runs = []

    def execute(notebook, stem, timeout):
        runs.append(stem)
        return [f"output {len(runs)}.{k}\n" for k in range(2)]

    monkeypatch.setattr(docs, "execute", execute)
    return script, runs


def test_text_change_reuses_the_outputs(example):
    script, runs = example
    docs.render(script, run=False)  # no page yet: runs
    script.write_text(SCRIPT.replace("Some text.", "Other text."), encoding="utf-8")
    docs.render(script, run=False)
    page = (script.parent / "example_test.md").read_text(encoding="utf-8")
    assert len(runs) == 1
    assert "Other text." in page and "output 1.1" in page
    assert docs.is_current(script)


def test_code_change_or_named_render_runs_again(example):
    script, runs = example
    docs.render(script, run=False)
    docs.render(script, run=True)  # a named render always runs
    assert len(runs) == 2
    script.write_text(SCRIPT.replace("x = 1", "x = 2"), encoding="utf-8")
    docs.render(script, run=False)
    assert len(runs) == 3
    assert "output 3.0" in (script.parent / "example_test.md").read_text(
        encoding="utf-8"
    )


def test_empty_outputs_survive_a_text_rebuild(example, monkeypatch):
    script, runs = example
    monkeypatch.setattr(docs, "execute", lambda *a: runs.append(1) or ["", ""])
    docs.render(script, run=False)
    script.write_text(SCRIPT + "\n# %% [markdown]\n# End.\n", encoding="utf-8")
    docs.render(script, run=False)
    assert len(runs) == 1
