import torch
import pytest

from gpt_from_scratch import LanguageModelDataset


def test_batch_targets_are_inputs_shifted_by_one_token() -> None:
    dataset = LanguageModelDataset.from_text("abcdefghijklmnopqrstuvwxyz", train_fraction=0.75)
    generator = torch.Generator().manual_seed(42)

    inputs, targets = dataset.get_batch(
        "train", batch_size=4, context_length=5, generator=generator
    )

    assert inputs.shape == (4, 5)
    assert targets.shape == (4, 5)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_batch_rejects_context_longer_than_split() -> None:
    dataset = LanguageModelDataset.from_text("abcdefghij", train_fraction=0.8)

    with pytest.raises(ValueError, match="must be longer"):
        dataset.get_batch("validation", batch_size=1, context_length=2)
