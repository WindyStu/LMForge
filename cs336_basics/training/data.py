import numpy as np
import numpy.typing as npt
import torch

def get_batch(
    dataset: npt.NDArray,
        batch_size: int,
        context_length: int,
        device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    random_positions = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    offset = np.arange(context_length)
    indices = random_positions[:, None] + offset[None, :]
    inputs = torch.tensor(
        dataset[indices],
        device=device,
        dtype=torch.long,
    )
    label = torch.tensor(
        dataset[indices + 1],
        device=device,
        dtype=torch.long,
    )

    return inputs, label