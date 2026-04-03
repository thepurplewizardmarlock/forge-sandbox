"""
hello.py — Starter script for Forge Sandbox.

A simple greeting function with input validation.
Run directly or import into other files.
"""


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
    if not isinstance(name, str):
        raise TypeError(f"Expected a string, got {type(name).__name__}")
    name = name.strip()
    if not name:
        raise ValueError("Name cannot be empty")
    return f"Hello, {name}!"


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
    if not isinstance(name, str):
        raise TypeError(f"Expected a string, got {type(name).__name__}")
    name = name.strip()
    if not name:
        raise ValueError("Name cannot be empty")
    return f"HELLO, {name.upper()}!"


if __name__ == "__main__":
    user_input = input("What's your name? ")
    print(greet(user_input))
