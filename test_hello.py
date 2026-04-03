"""
test_hello.py — Tests for hello.py

Run with:
    python3 test_hello.py
    python3 -m pytest test_hello.py -v
"""

import unittest
from hello import greet, shout


class TestGreet(unittest.TestCase):
    """Tests for the greet() function."""

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


if __name__ == "__main__":
    unittest.main()
