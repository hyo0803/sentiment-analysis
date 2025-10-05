# tests/test_preprocess.py
import pytest
from src.data.utils import preprocess_text

def test_preprocess_text_removes_newlines_and_tabs():
    input_text = "Hello\n\tWorld\r\n!"
    expected = "Hello World !"
    assert preprocess_text(input_text) == expected

def test_preprocess_text_handles_multiple_spaces():
    input_text = "Too    many     spaces"
    expected = "Too many spaces"
    assert preprocess_text(input_text) == expected

def test_preprocess_text_strips_whitespace():
    input_text = "  leading and trailing  "
    expected = "leading and trailing"
    assert preprocess_text(input_text) == expected

def test_preprocess_text_converts_non_string_to_string():
    assert preprocess_text(123) == "123"
    assert preprocess_text(None) == "None"

def test_preprocess_text_empty_string():
    assert preprocess_text("") == ""
    assert preprocess_text("   ") == ""

def test_preprocess_text_preserves_letters_and_punctuation():
    input_text = "It's great! 😊 (but expensive...)"
    # Эмодзи остаются, но управляющие символы — нет
    assert "It's great! 😊 (but expensive...)" == preprocess_text(input_text)