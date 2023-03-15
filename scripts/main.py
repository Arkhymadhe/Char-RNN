import json
import os
import shutil

import torch
from torch import nn, optim

from data_ops import get_text, encode_text, get_char_mapping, batch_sequence, split_data
from scripts.inference_ops import text_predict, prime_model
from scripts.net import CharRNN, get_base_rnn
from scripts.train_ops import train_model


def main(op):
    device = 'cuda'

    if op.lower() == "train":
        path = os.path.join(
            os.getcwd().replace("scripts", "data"), "anna.txt"
        )
        data = get_text(path)

        train_data, val_data = split_data(data, train_frac=.8)

        unique_chars = list(set(data))

        train_data, unique_chars, int2char, char2int = encode_text(train_data, False, unique_chars=unique_chars)

        val_data, unique_chars, int2char, char2int = encode_text(val_data, False, unique_chars)

        chars2int = {char: unique_chars.index(char) for char in unique_chars}
        int2char = {v: k for (k, v) in chars2int.items()}

        print("chars2int: ", len(chars2int))

        print("unique_chars: ", len(unique_chars))

        batch_size = 128
        bi_directional = 1
        dropout=0.5
        seq_length = 100
        num_layers = 2
        hidden_size = 512

        max_norm = 15
        epochs = 30
        lr = 1e-3
        base_rnn_name = 'lstm'
        base_rnn = get_base_rnn(base_rnn_name)
        vocab_size = len(unique_chars)

        model = CharRNN(
            D=bi_directional, dropout=dropout, num_layers=num_layers, base_rnn=base_rnn,
            batch_size=batch_size, hidden_size=hidden_size,
            input_size=vocab_size, output_size=vocab_size
        ).to(device)

        ### Objective functions and optimizer
        opt = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        model, opt = train_model(
            model,
            opt,
            train_data,
            val_data,
            criterion,
            epochs,
            batch_size,
            seq_length,
            max_norm,
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
            batch_size=batch_size,
            D=bi_directional,
            dropout=dropout,
            num_layers=num_layers,
            hidden_size=hidden_size,
            base_rnn=base_rnn_name,
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
        pass
    elif op.lower() == "inference":
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
            "Generated Text:", "\n", "=="*20, "\n", "".join(generated_seed_list)
        )
        pass
    else:
        print("Only train and inference operations allowed!")
    return


if __name__ == "__main__":
    op = 'inference'
    main(op)
