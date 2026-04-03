"""
test_hello.py — Tests for hello.py

Run with:
    python3 test_hello.py
    python3 -m pytest test_hello.py -v
"""

import unittest
from unittest.mock import patch
from hello import greet, shout, whisper, _normalize, main


class TestNormalize(unittest.TestCase):
    """Tests for the _normalize() helper."""

    def test_returns_clean_string(self):
        self.assertEqual(_normalize("Master"), "Master")

    def test_strips_edges(self):
        self.assertEqual(_normalize("  Master  "), "Master")

    def test_collapses_internal_whitespace(self):
        self.assertEqual(_normalize("John  Smith"), "John Smith")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            _normalize("")

    def test_whitespace_only_raises(self):
        with self.assertRaises(ValueError):
            _normalize("   ")

    def test_non_string_raises(self):
        with self.assertRaises(TypeError):
            _normalize(42)

    def test_none_raises(self):
        with self.assertRaises(TypeError):
            _normalize(None)


class TestGreet(unittest.TestCase):
    """Tests for the greet() function."""

    def test_default_name(self):
        self.assertEqual(greet(), "Hello, World!")

    def test_basic_greeting(self):
        self.assertEqual(greet("Master"), "Hello, Master!")

    def test_strips_whitespace(self):
        self.assertEqual(greet("  Master  "), "Hello, Master!")

    def test_normalizes_internal_whitespace(self):
        # Double space between words should collapse to one
        self.assertEqual(greet("John  Smith"), "Hello, John Smith!")

    def test_empty_string_raises(self):
        with self.assertRaises(ValueError):
            greet("")

    def test_whitespace_only_raises(self):
        with self.assertRaises(ValueError):
            greet("   ")

    def test_non_string_raises(self):
        with self.assertRaises(TypeError):
            greet(42)

    def test_none_raises(self):
        with self.assertRaises(TypeError):
            greet(None)


class TestShout(unittest.TestCase):
    """Tests for the shout() function."""

    def test_default_name(self):
        self.assertEqual(shout(), "HELLO, WORLD!")

    def test_basic_shout(self):
        self.assertEqual(shout("Master"), "HELLO, MASTER!")

    def test_strips_whitespace(self):
        self.assertEqual(shout("  Master  "), "HELLO, MASTER!")

    def test_empty_string_raises(self):
        with self.assertRaises(ValueError):
            shout("")

    def test_whitespace_only_raises(self):
        with self.assertRaises(ValueError):
            shout("   ")

    def test_non_string_raises(self):
        with self.assertRaises(TypeError):
            shout(42)

    def test_none_raises(self):
        with self.assertRaises(TypeError):
            shout(None)


class TestWhisper(unittest.TestCase):
    """Tests for the whisper() function."""

    def test_default_name(self):
        self.assertEqual(whisper(), "hello, world!")

    def test_basic_whisper(self):
        self.assertEqual(whisper("Master"), "hello, master!")

    def test_strips_whitespace(self):
        self.assertEqual(whisper("  Master  "), "hello, master!")

    def test_empty_string_raises(self):
        with self.assertRaises(ValueError):
            whisper("")

    def test_whitespace_only_raises(self):
        with self.assertRaises(ValueError):
            whisper("   ")

    def test_non_string_raises(self):
        with self.assertRaises(TypeError):
            whisper(42)

    def test_none_raises(self):
        with self.assertRaises(TypeError):
            whisper(None)


class TestMain(unittest.TestCase):
    """Tests for the interactive main() function."""

    def test_valid_name_prints_greeting(self):
        with patch("builtins.input", return_value="Master"), \
             patch("builtins.print") as mock_print:
            main()
            mock_print.assert_called_once_with("Hello, Master!")

    def test_empty_input_does_not_crash(self):
        # User hits Enter with no input — should print an error, not raise
        with patch("builtins.input", return_value=""), \
             patch("builtins.print") as mock_print:
            main()  # must not raise
            args = mock_print.call_args[0][0]
            self.assertIn("Error", args)


if __name__ == "__main__":
    unittest.main()
