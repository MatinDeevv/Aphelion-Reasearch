"""
Tests for APHELION Phase 23 — Widgets.

Covers TextArea, ProgressBar, Modal, FormField, and FilePicker widgets.
"""

from __future__ import annotations

import os
import tempfile

import pytest

rich = pytest.importorskip("rich")

from rich.panel import Panel
from rich.text import Text


# ═══════════════════════════════════════════════════════════════════════════
# TextArea
# ═══════════════════════════════════════════════════════════════════════════

from aphelion.tui.widgets.text_area import TextAreaState, render_text_area


class TestTextAreaState:
    def test_default_empty(self):
        ta = TextAreaState()
        assert ta.content == ""
        assert ta.line_count == 1
        assert ta.is_empty is True

    def test_insert_char(self):
        ta = TextAreaState()
        ta.insert_char("A")
        ta.insert_char("B")
        assert ta.content == "AB"
        assert ta.cursor_col == 2

    def test_insert_newline(self):
        ta = TextAreaState()
        ta.insert_char("A")
        ta.insert_newline()
        ta.insert_char("B")
        assert ta.line_count == 2
        assert ta.lines[0] == "A"
        assert ta.lines[1] == "B"

    def test_backspace(self):
        ta = TextAreaState()
        ta.insert_char("A")
        ta.insert_char("B")
        ta.backspace()
        assert ta.content == "A"

    def test_backspace_at_line_start(self):
        ta = TextAreaState()
        ta.insert_char("A")
        ta.insert_newline()
        ta.insert_char("B")
        ta.backspace()
        ta.backspace()  # merges with previous line
        assert ta.line_count == 1

    def test_clear(self):
        ta = TextAreaState()
        ta.insert_char("X")
        ta.insert_newline()
        ta.insert_char("Y")
        ta.clear()
        assert ta.content == ""
        assert ta.line_count == 1

    def test_paste(self):
        ta = TextAreaState()
        ta.paste("hello\nworld")
        assert ta.line_count == 2
        assert "hello" in ta.content
        assert "world" in ta.content

    def test_content_setter(self):
        ta = TextAreaState()
        ta.content = "line1\nline2\nline3"
        assert ta.line_count == 3
        assert ta.lines == ["line1", "line2", "line3"]

    def test_max_lines_check(self):
        ta = TextAreaState(max_lines=3)
        ta.content = "a\nb\nc\nd\ne"
        assert ta.line_count <= 3


class TestRenderTextArea:
    def test_renders_panel(self):
        ta = TextAreaState()
        ta.insert_char("Hello")
        panel = render_text_area(ta, title="Test")
        assert isinstance(panel, Panel)

    def test_empty_renders(self):
        ta = TextAreaState()
        panel = render_text_area(ta)
        assert panel is not None


# ═══════════════════════════════════════════════════════════════════════════
# ProgressBar
# ═══════════════════════════════════════════════════════════════════════════

from aphelion.tui.widgets.progress_bar import (
    render_progress_bar,
    render_epoch_progress,
    render_loss_sparkline,
)


class TestProgressBar:
    def test_render_zero(self):
        txt = render_progress_bar("Test", 0.0)
        assert isinstance(txt, Text)

    def test_render_full(self):
        txt = render_progress_bar("Done", 1.0)
        assert isinstance(txt, Text)

    def test_render_with_elapsed(self):
        txt = render_progress_bar("Work", 0.5, message="halfway", elapsed=120)
        rendered = str(txt)
        assert "50%" in rendered

    def test_clamps_percent(self):
        txt = render_progress_bar("Over", 1.5)
        rendered = str(txt)
        assert "100%" in rendered

    def test_negative_percent(self):
        txt = render_progress_bar("Under", -0.5)
        rendered = str(txt)
        assert "0%" in rendered


class TestEpochProgress:
    def test_render(self):
        txt = render_epoch_progress(5, 20)
        assert isinstance(txt, Text)

    def test_zero_total(self):
        txt = render_epoch_progress(0, 0)
        assert isinstance(txt, Text)


class TestLossSparkline:
    def test_renders_with_data(self):
        txt = render_loss_sparkline([0.5, 0.4, 0.3, 0.2, 0.15, 0.12])
        assert isinstance(txt, Text)

    def test_renders_empty(self):
        txt = render_loss_sparkline([])
        assert isinstance(txt, Text)


# ═══════════════════════════════════════════════════════════════════════════
# Modal
# ═══════════════════════════════════════════════════════════════════════════

from aphelion.tui.widgets.modal import (
    ModalState,
    ModalButton,
    render_modal,
    build_quit_confirmation_modal,
    build_forge_success_modal,
    build_forge_failure_modal,
    build_first_run_modal,
)


class TestModalState:
    def test_default_hidden(self):
        ms = ModalState()
        assert ms.visible is False
        assert ms.result is None

    def test_show(self):
        ms = ModalState()
        ms.show(
            "Confirm",
            ["Are you sure?"],
            [ModalButton(key="Y", label="Yes", action="confirm")],
        )
        assert ms.visible is True
        assert ms.title == "Confirm"
        assert len(ms.buttons) == 1

    def test_dismiss(self):
        ms = ModalState()
        ms.show("Test", ["body"], [])
        ms.dismiss("cancel")
        assert ms.visible is False
        assert ms.result == "cancel"


class TestModalRender:
    def test_render_visible(self):
        ms = ModalState()
        ms.show("Title", ["Body line"], [ModalButton("Y", "Yes", "ok")])
        panel = render_modal(ms)
        assert isinstance(panel, Panel)

    def test_render_hidden_returns_empty(self):
        ms = ModalState()
        result = render_modal(ms)
        # Should return Text("") or Panel when not visible — either is valid
        assert result is not None


class TestPrebuiltModals:
    def test_quit_confirmation(self):
        ms = build_quit_confirmation_modal()
        assert ms.visible is True
        assert any("stop" in b.action.lower() or "cancel" in b.action.lower() for b in ms.buttons)

    def test_forge_success(self):
        ms = build_forge_success_modal("RSI_Cross", sharpe=1.8)
        assert ms.visible is True

    def test_forge_failure(self):
        ms = build_forge_failure_modal("Bad_Strat", ["Sharpe < 0"])
        assert ms.visible is True

    def test_first_run(self):
        ms = build_first_run_modal()
        assert ms.visible is True


# ═══════════════════════════════════════════════════════════════════════════
# FormField
# ═══════════════════════════════════════════════════════════════════════════

from aphelion.tui.widgets.form_field import FormField, FormState, render_form


class TestFormField:
    def test_defaults(self):
        ff = FormField(key="test.key", label="Test")
        assert ff.value == ""
        assert ff.field_type == "text"

    def test_set_value_clears_error(self):
        ff = FormField(key="k", label="L", validation_error="bad")
        ff.set_value("new_value")
        assert ff.value == "new_value"
        assert ff.validation_error == ""

    def test_display_value_password(self):
        ff = FormField(key="k", label="L", field_type="password", value="secret123")
        assert "•" in ff.display_value
        assert "secret" not in ff.display_value

    def test_display_value_placeholder(self):
        ff = FormField(key="k", label="L", placeholder="enter...")
        assert ff.display_value == "enter..."

    def test_validate_number_valid(self):
        ff = FormField(key="k", label="L", field_type="number", value="42")
        assert ff.validate() is True

    def test_validate_number_invalid(self):
        ff = FormField(key="k", label="L", field_type="number", value="abc")
        assert ff.validate() is False
        assert ff.validation_error != ""

    def test_validate_select_valid(self):
        ff = FormField(key="k", label="L", field_type="select",
                       options=["a", "b", "c"], value="b")
        assert ff.validate() is True

    def test_validate_select_invalid(self):
        ff = FormField(key="k", label="L", field_type="select",
                       options=["a", "b", "c"], value="d")
        assert ff.validate() is False


class TestFormState:
    def _make_form(self):
        return FormState(fields=[
            FormField(key="f1", label="First", value="a", section="S1"),
            FormField(key="f2", label="Second", value="b", section="S1"),
            FormField(key="f3", label="Third", value="c", section="S2"),
        ])

    def test_active_field(self):
        form = self._make_form()
        assert form.active_field.key == "f1"

    def test_next_prev_field(self):
        form = self._make_form()
        form.next_field()
        assert form.active_field.key == "f2"
        form.next_field()
        assert form.active_field.key == "f3"
        form.next_field()  # wraps
        assert form.active_field.key == "f1"
        form.prev_field()  # wraps back
        assert form.active_field.key == "f3"

    def test_get_set_value(self):
        form = self._make_form()
        form.set_value("f2", "new")
        assert form.get_value("f2") == "new"
        assert form.dirty is True

    def test_validate_all(self):
        form = FormState(fields=[
            FormField(key="k1", label="Num", field_type="number", value="abc"),
            FormField(key="k2", label="Good", value="ok"),
        ])
        errors = form.validate_all()
        assert len(errors) == 1
        assert "Num" in errors[0]

    def test_to_dict(self):
        form = self._make_form()
        d = form.to_dict()
        assert d == {"f1": "a", "f2": "b", "f3": "c"}

    def test_sections(self):
        form = self._make_form()
        assert form.sections == ["S1", "S2"]


class TestRenderForm:
    def test_renders_table(self):
        form = FormState(fields=[
            FormField(key="k", label="Field", value="val", section="Sec"),
        ])
        from rich.table import Table
        table = render_form(form)
        assert isinstance(table, Table)


# ═══════════════════════════════════════════════════════════════════════════
# FilePicker
# ═══════════════════════════════════════════════════════════════════════════

from aphelion.tui.widgets.file_picker import FilePickerState, render_file_picker


class TestFilePickerState:
    def test_default_hidden(self):
        fp = FilePickerState()
        assert fp.visible is False

    def test_open(self, tmp_path):
        fp = FilePickerState()
        fp.open(str(tmp_path))
        assert fp.visible is True
        assert ".." in fp.entries

    def test_navigate(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "test.txt").write_text("x")

        fp = FilePickerState()
        fp.navigate_to(str(tmp_path))
        assert "subdir/" in fp.entries
        assert "test.txt" in fp.entries

    def test_filter_ext(self, tmp_path):
        (tmp_path / "a.csv").write_text("x")
        (tmp_path / "b.pt").write_text("x")

        fp = FilePickerState()
        fp.open(str(tmp_path), filter_ext=".pt")
        assert "b.pt" in fp.entries
        assert "a.csv" not in fp.entries

    def test_move_up_down(self, tmp_path):
        (tmp_path / "a").write_text("x")
        (tmp_path / "b").write_text("x")

        fp = FilePickerState()
        fp.open(str(tmp_path))
        fp.move_down()
        assert fp.selected_index == 1
        fp.move_up()
        assert fp.selected_index == 0

    def test_select_file(self, tmp_path):
        (tmp_path / "data.csv").write_text("x")
        fp = FilePickerState()
        fp.open(str(tmp_path))
        # Move to "data.csv" entry
        for i, entry in enumerate(fp.entries):
            if entry == "data.csv":
                fp.selected_index = i
                break
        fp.select()
        assert fp.selected_path is not None
        assert "data.csv" in fp.selected_path

    def test_cancel(self, tmp_path):
        fp = FilePickerState()
        fp.open(str(tmp_path))
        fp.cancel()
        assert fp.visible is False


class TestRenderFilePicker:
    def test_renders_panel(self, tmp_path):
        fp = FilePickerState()
        fp.open(str(tmp_path))
        panel = render_file_picker(fp)
        assert isinstance(panel, Panel)

    def test_renders_hidden(self):
        fp = FilePickerState()
        result = render_file_picker(fp)
        assert result is not None
