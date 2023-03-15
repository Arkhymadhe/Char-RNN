from torch.nn import functional as F
from torch import from_numpy, nn, distributions
from numpy.random import choice
from numpy import array
from data_ops import one_hot_encode

from typing import Tuple, Union

from torch import Tensor


def character_predict(x, h, net, char2int, code_size=83, k=5):
    device = next(net.parameters()).device
    x = array([[char2int[x]]])
    x = one_hot_encode(x, code_size)
    x = from_numpy(x).to(device).contiguous()

    out, h = net(x, h)
    p = F.softmax(out, dim=-1).data
    p, chars = p.topk(k, dim=-1)

    chars = chars.detach().cpu().numpy().squeeze()
    p = p.detach().cpu().numpy().squeeze()

    return choice(chars, p=p / p.sum()), h


def text_predict(
    model,
    int2char: dict,
    seed_list: list,
    num_chars: int,
    k: int,
    h: Union[Tuple[Tensor, Tensor], Tensor],
) -> [str, Tensor]:

    char2int = {v : k for (k, v) in int2char.items()}
    code_size = len(char2int)

    for n in range(num_chars):
        h = tuple([each.data for each in h])
        next_char, h = character_predict(seed_list[-1], h, model, char2int, code_size, k=k)
        seed_list.append(int2char[next_char])

    return "".join(seed_list), h


def prime_model(model, seed, k, int2char):

    device = next(model.parameters()).device
    num_layers = model.num_layers
    hidden_size = model.hidden_size
    seed_list = list(seed)

    char2int = {v: k for (k, v) in int2char.items()}
    code_size = len(char2int)

    h = distributions.Normal(scale = 0.5, loc = 0.).sample((num_layers, 1, hidden_size))
    if model.base_rnn == nn.LSTM:
        b = distributions.Normal(scale = 0.5, loc = 0.).sample((num_layers, 1, hidden_size))
        h = (h.to(device), b.to(device))

    for char in seed_list:
        h = tuple([each.data for each in h]) if type(h) == tuple else h.data
        next_char, h = character_predict(char, h, model, char2int, code_size, k=k)

    seed_list.append(int2char[next_char])

    return seed_list, h
