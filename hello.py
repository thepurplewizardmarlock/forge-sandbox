"""
hello.py — Starter script for Forge Sandbox.

A simple greeting function with input validation.
Run directly or import into other files.
"""


def _normalize(name: str) -> str:
    """Validate and normalize a name string.

    Raises TypeError for non-strings, ValueError for blank input.
    Returns the name with leading/trailing/internal whitespace collapsed.
    """
    if not isinstance(name, str):
        raise TypeError(f"Expected a string, got {type(name).__name__}")
    name = " ".join(name.split())
    if not name:
        raise ValueError("Name cannot be empty")
    return name


def greet(name: str) -> str:
    """Return a greeting for the given name.

    Args:
        name: The name to greet. Must be a non-empty string.

    Returns:
        A greeting string.

    Raises:
        ValueError: If name is empty or whitespace-only.
        TypeError: If name is not a string.
    """
    return f"Hello, {_normalize(name)}!"


def shout(name: str) -> str:
    """Return an uppercased greeting for the given name.

    Args:
        name: The name to greet. Must be a non-empty string.

    Returns:
        An uppercased greeting string.

    Raises:
        ValueError: If name is empty or whitespace-only.
        TypeError: If name is not a string.
    """
    return f"HELLO, {_normalize(name).upper()}!"


if __name__ == "__main__":
    user_input = input("What's your name? ")
    print(greet(user_input))
