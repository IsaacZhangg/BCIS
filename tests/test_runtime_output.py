"""Tests for runtime_output helpers."""

from src.runtime_output import _LINE_WIDTH, _print_header, _print_step


def test_print_header_outputs_title(capsys):
    _print_header("Hello")
    captured = capsys.readouterr()
    assert "Hello" in captured.out
    assert "=" * _LINE_WIDTH in captured.out


def test_print_step_shows_index_and_title(capsys):
    _print_step(2, 5, "Doing thing")
    captured = capsys.readouterr()
    assert "[2/5]" in captured.out
    assert "Doing thing" in captured.out
