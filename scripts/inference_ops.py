from torch.nn import functional as F
from torch import from_numpy
from numpy.random import choice
from numpy import array
from data_ops import one_hot_encode

from typing import Optional, Tuple

from torch import Tensor


def character_predict(x, h, net, char2int, code_size = 83, device = "cuda", k=5):
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
    h: Optional[Tuple[Tensor, Tensor], Tensor],
) -> str:

    for n in range(num_chars):
        h = tuple([each.data for each in h])
        next_char, h = character_predict(seed_list[-1], h, model, k=k)
        seed_list.append(int2char[next_char])

    return "".join(seed_list)
