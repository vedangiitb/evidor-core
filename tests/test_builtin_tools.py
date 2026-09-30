from datetime import datetime, timedelta

import pytest

from evidor import calculator, get_current_time
from evidor.tools import Tool


def test_builtin_tools_are_tool_definitions() -> None:
    assert isinstance(calculator, Tool)
    assert isinstance(get_current_time, Tool)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(12 + 3) * 4", 60),
        ("-8 / 2", -4),
        ("2 ** 3 + 5 % 3", 10),
        ("7 // 2", 3),
    ],
)
def test_calculator_supported_arithmetic(expression: str, expected: int | float) -> None:
    assert calculator(expression=expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo unsafe')",
        "2 ** 101",
        "1 / 0",
        "x + 1",
        "1 + " + "1" * 256,
    ],
)
def test_calculator_rejects_unsupported_or_unsafe_input(expression: str) -> None:
    with pytest.raises(ValueError):
        calculator(expression=expression)


def test_get_current_time_returns_iso_time_in_requested_timezone() -> None:
    value = get_current_time(timezone="UTC")
    parsed = datetime.fromisoformat(value)

    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)


def test_get_current_time_rejects_unknown_timezone() -> None:
    with pytest.raises(ValueError, match="Unknown timezone"):
        get_current_time(timezone="Not/A_Timezone")
