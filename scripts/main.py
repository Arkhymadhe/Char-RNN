import json
import os
import shutil
from typing import Union

import torch
from torch import nn, optim

from data_ops import get_text, encode_text, split_data
from inference_ops import text_predict, prime_model
from net import CharRNN, get_base_rnn
from train_ops import train_model

from argparse import ArgumentParser


def main():
    args = ArgumentParser()
    args = parse_args(args)
    args = args.parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    if args.op.lower() == "train":
        path = os.path.join(
            os.getcwd().replace("scripts", "data"), "anna.txt"
        )
        data = get_text(path)

        train_data, val_data = split_data(data, train_frac=args.train_frac)

        unique_chars = list(set(data))

        train_data, unique_chars, int2char, char2int = encode_text(train_data, False, unique_chars=unique_chars)

        val_data, unique_chars, int2char, char2int = encode_text(val_data, False, unique_chars)

        chars2int = {char: unique_chars.index(char) for char in unique_chars}
        int2char = {v: k for (k, v) in chars2int.items()}

        print("chars2int: ", len(chars2int))

        print("unique_chars: ", len(unique_chars))

        vocab_size = len(unique_chars)

        model = CharRNN(
            D=args.bi_directional, dropout=args.dropout, num_layers=args.num_layers, base_rnn=get_base_rnn(args.base),
            batch_size=args.batch_size, hidden_size=args.hidden_size,
            input_size=vocab_size, output_size=vocab_size
        ).to(device)

        ### Objective functions and optimizer
        opt = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        model, opt = train_model(
            model,
            opt,
            train_data,
            val_data,
            criterion,
            args.epochs,
            args.batch_size,
            args.seq_length,
            args.max_norm,
            device,
            code_size=len(unique_chars)
        )

        print(
            "Char-RNN fully trained!"
        )
        with open('latest-weights.net', 'wb') as f:
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        }, f)

        src = os.path.join(os.getcwd(), 'latest-weights.net')
        dst = os.path.join(os.getcwd().replace("scripts", "artefacts"), 'latest-weights.net')

        shutil.move(src, dst)

        with open(
                os.path.join(
                    os.getcwd().replace("scripts", "artefacts"), "int2char.json"
                ), "w") as f:
            json.dump(int2char, f, indent=4)

        model_params = dict(
            batch_size=args.batch_size,
            D=args.bi_directional,
            dropout=args.dropout,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            base_rnn=args.base,
            input_size=vocab_size,
            output_size=vocab_size
        )

        with open(
                os.path.join(
                    os.getcwd().replace("scripts", "artefacts"), "model_params.json"
                ), "w") as f:
            json.dump(model_params, f, indent=4)

        print(
            "Model persisted!"
        )

    elif args.op.lower() == "inference":
        seeds = [
            """Lan al'Mandragoran went up the mountain""",
            """Nevermore, nevermore, the raven said""",
            """To be or not to be. That is the main question""",
            """Why is water wet and fire hot?"""
        ]

        seed = seeds[torch.randint(low=0, high=len(seeds), size=(1,)).item()]
        k = 5
        num_chars = 1000

        with open(
                os.path.join(
                    os.getcwd().replace("scripts", "artefacts"), "int2char.json"
                ), "r") as f:
            int2char = json.load(f)
            int2char = {int(k) : v for (k, v) in int2char.items()}

        with open(
                os.path.join(
                    os.getcwd().replace("scripts", "artefacts"), "model_params.json"
                ), "r") as f:
            params = json.load(f)

        print(params)
        params['base_rnn'] = get_base_rnn(params['base_rnn'])

        model = CharRNN(**params).to(device)

        file = os.path.join(os.getcwd().replace("scripts", "artefacts"), 'latest-weights.net')

        with open(file, 'rb') as f:
            state = torch.load(f)

        model.load_state_dict(state['model_state_dict'])
        model.eval()

        print(
            "Loaded saved model!"
        )

        generated_seed_list, h = prime_model(model, seed, k, int2char)

        generated_seed_list, h = text_predict(
            model,
            int2char,
            generated_seed_list,
            num_chars,
            k,
            h
        )

        print(
            "Generated Text:", "=="*20, "".join(generated_seed_list), sep='\n'
        )

    else:
        print("Only train and inference operations allowed!")
    return


def parse_args(args):
    args.add_argument(
        "--op",
        choices=['train', 'inference'],
        default='train',
        type=str
    )
    args.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )

    args.add_argument(
        "--bi_directional",
        type=int,
        default=1,
        choices=[0, 1]
    )

    args.add_argument(
        "--dropout",
        default=0.5,
        type=float,
    )

    args.add_argument(
        "--seq_length",
        type=int,
        default=100,
    )

    args.add_argument(
        "--num_layers",
        type=int,
        default=2
    )

    args.add_argument(
        "--hidden_size",
        default=512,
        type=int
    )

    args.add_argument(
        "--max_norm",
        default=15,
        type=Union[float, int]
    )

    args.add_argument(
        "--lr",
        default=1e-3,
        type=float
    )

    args.add_argument(
        "--epochs",
        default=30,
        type=int
    )

    args.add_argument(
        "--base",
        type=str,
        default="lstm",
        choices=['lstm', 'gru', 'rnn']
    )

    args.add_argument(
        "--train_frac",
        default=.8,
        type=float,
    )

    print("Added all arguments")

    return args


if __name__ == "__main__":

    main()
