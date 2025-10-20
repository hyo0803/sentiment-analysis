# tests/test_preprocess.py
import pytest
import numpy as np # type: ignore
import argparse
from src.data.utils import preprocess_text, to_python_ints, parse_label_mapping

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
    assert preprocess_text(input_text) == "It's great! 😊 (but expensive...)"

# New tests for to_python_ints
def test_to_python_ints_scalar_and_array_and_numpy_scalar():
    assert to_python_ints(np.int64(0)) == 0
    assert to_python_ints([np.int64(0)]) == [0]
    assert to_python_ints(np.array([np.int64(0)])) == [0]
    assert to_python_ints(5) == 5
    assert to_python_ints([1, 2, np.int64(3)]) == [1, 2, 3]

# New tests for parse_label_mapping
def test_parse_label_mapping_accepts_json_and_python_literal_and_dict():
    assert parse_label_mapping('{"-1": 0, "1": 1}') == {-1: 0, 1: 1}
    assert parse_label_mapping("{-1: 0, 1: 1}") == {-1: 0, 1: 1}
    assert parse_label_mapping({-1: 0, 1: 1}) == {-1: 0, 1: 1}
    # string numeric keys become ints
    assert parse_label_mapping('{"0":"neg","1":"pos"}') == {0: "neg", 1: "pos"}

def test_parse_label_mapping_invalid_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_label_mapping("not a mapping")