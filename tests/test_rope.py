import pytest
import torch

from mla_decode import MLAConfig
from mla_decode.rope import apply_rope


def test_rope_rotates_each_request_at_its_own_position() -> None:
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]], [[1.0, 2.0, 3.0, 4.0]]])
    actual = apply_rope(x, torch.tensor([0, 1]))
    torch.testing.assert_close(actual[0], x[0])
    one = torch.tensor(1.0)
    angle = torch.tensor(0.01)
    expected_first = torch.tensor(
        [one.cos() - 3 * one.sin(), 2 * angle.cos() - 4 * angle.sin()]
    )
    expected_second = torch.tensor(
        [one.sin() + 3 * one.cos(), 2 * angle.sin() + 4 * angle.cos()]
    )
    torch.testing.assert_close(actual[1, 0, :2], expected_first)
    torch.testing.assert_close(actual[1, 0, 2:], expected_second)


def test_rope_dimension_is_even() -> None:
    with pytest.raises(ValueError, match="even"):
        MLAConfig(8, 2, 4, 4, 4, 3, 4)
