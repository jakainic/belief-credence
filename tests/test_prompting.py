"""Tests for direct prompting method."""

from belief_credence.prompting import DirectPrompting


def test_parse_credence_valid() -> None:
    method = DirectPrompting(model_name="test")

    assert method._parse_credence("0.75") == 0.75
    assert method._parse_credence("The answer is 0.5") == 0.5
    assert method._parse_credence("1.0") == 1.0
    assert method._parse_credence("0.0") == 0.0
    assert method._parse_credence("My credence is .85 based on...") == 0.85


def test_parse_credence_invalid() -> None:
    method = DirectPrompting(model_name="test")

    import pytest

    with pytest.raises(ValueError, match="Could not parse"):
        method._parse_credence("invalid")

    with pytest.raises(ValueError, match="Could not parse"):
        method._parse_credence("1.5")

    with pytest.raises(ValueError, match="Could not parse"):
        method._parse_credence("-0.5")
