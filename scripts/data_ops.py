import PyPDF3
import numpy as np
from collections import namedtuple
from typing import Tuple


def get_text(fpath):
    if fpath.lower().endswith(".pdf"):
        text = load_pdf(fpath)

    elif fpath.lower().endswith(".txt"):
        with open(fpath, "r") as f:
            text = f.read()

    return text


def load_pdf(fpath):
    with open(fpath, "rb") as f:
        pdf = PyPDF3.PdfFileReader(f)
        text = str()
        for page_num in range(pdf.numPages):
            page = pdf.getPage(page_num)
            text = text + " " + page.extractText()

    return text


def encode_text(
    text: str, extend: bool = True, unique_chars: list = None
) -> Tuple[np.ndarray, list, dict, dict]:
    """
    Takes in a piece of text and encodes the characters of the text by unique numerical identifiers.

    Parameters
    ----------
    text
        Text to be encoded.
    extend
        Expands set of possible unique characters.
    unique_chars
        Set of unique characters

    Returns
    -------
    out
        A Tuple containing the encoded text, the set of unique characters, mapping from numerical code to character,
        and mapping from character to numerical code.

    """
    result_tuple = namedtuple(
        "results", ["encoded_text", "unique_char", "int2char", "char2int"]
    )

    if unique_chars is None:
        unique_chars = list(set(text).union(set("#[]{}+-*=!")))
    if extend:
        unique_chars.extend(list("#[]{}+-*=!"))

    char2int = {char: unique_chars.index(char) for char in unique_chars}
    int2char = {v: k for (k, v) in char2int.items()}

    encoded_text = np.array(list(map(lambda x: char2int[x], list(text))))

    return result_tuple(encoded_text, unique_chars, int2char, char2int)


def get_char_mapping(text, unique_chars=None):
    unique_chars = set() if not bool(unique_chars) else set(unique_chars)
    unique_chars.update(set(text))

    unique_chars = list(unique_chars)
    chars2int = {char: unique_chars.index(char) for char in unique_chars}
    int2char = {v: k for (k, v) in chars2int.items()}

    return chars2int, int2char, unique_chars


def split_data(data, train_frac=.8):
    train_data, val_data = data[:int(len(data) * train_frac)], data[int(len(data) * (train_frac)):]

    return train_data, val_data


def batch_sequence(arr, batch_size, seq_length):
    """
    Generate batches from encoded text.
    """
    numel_seq = batch_size * seq_length
    num_batches = arr.size // numel_seq

    arr = arr[: num_batches * numel_seq].reshape(batch_size, -1)
    # print(arr.shape)

    batched_data = [
        (arr[:, n : n + seq_length], arr[:, n + 1 : n + 1 + seq_length])
        for n in range(0, arr.shape[1], seq_length)
    ]

    ### Finalize final array size
    batched_data[-1] = (
        batched_data[-1][0],
        np.append(batched_data[-1][1], batched_data[0][1][:, 0].reshape(-1, 1), axis=1),
    )

    ###batched_arr = [arr[n : n + numel_seq].reshape(batch_size, seq_length) for n in range(num_batches)]
    return iter(batched_data), num_batches


def one_hot_encode(arr, n_labels):
    """
    Convert label encoding to one-hot code.
    """

    # Initialize the encoded array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)

    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.0

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot
