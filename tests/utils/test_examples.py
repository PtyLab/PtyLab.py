import pytest

examples = pytest.importorskip("PtyLab.utils.examples")

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
    """A one-example docs folder; `runs` counts how often the script is executed."""
    monkeypatch.setattr(examples, "PAGES", tmp_path)
    monkeypatch.setattr(examples, "SOURCES", tmp_path / "scripts")
    monkeypatch.setattr(examples, "OUTPUTS", tmp_path / "outputs")
    (tmp_path / "scripts").mkdir()
    script = tmp_path / "scripts" / "example_test.py"
    script.write_text(SCRIPT, encoding="utf-8")
    runs = []

    def execute(notebook, stem, timeout):
        runs.append(stem)
        return [f"output {len(runs)}.{k}\n" for k in range(2)]

    monkeypatch.setattr(examples, "execute", execute)
    return script, runs


def page(script):
    return (examples.PAGES / f"{script.stem}.md").read_text(encoding="utf-8")


def test_text_change_needs_no_run(example):
    script, runs = example
    assert not examples.build(script)  # no outputs yet
    examples.render(script)
    script.write_text(SCRIPT.replace("Some text.", "Other text."), encoding="utf-8")
    assert examples.build(script)
    assert len(runs) == 1
    assert "Other text." in page(script) and "output 1.1" in page(script)
    assert examples.is_current(script)


def test_code_change_needs_a_new_render(example):
    script, runs = example
    examples.render(script)
    script.write_text(SCRIPT.replace("x = 1", "x = 2"), encoding="utf-8")
    assert not examples.is_current(script) and not examples.build(script)
    examples.render(script)
    assert examples.build(script)
    assert "output 2.0" in page(script)


def test_empty_outputs_leave_no_gap(example, monkeypatch):
    script, runs = example
    monkeypatch.setattr(examples, "execute", lambda *a: ["", ""])
    examples.render(script)
    assert examples.build(script)
    assert "x = 1\n```\n\nSome text." in page(script)
