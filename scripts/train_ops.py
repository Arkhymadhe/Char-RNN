import torch
from matplotlib import pyplot as plt

from data_ops import one_hot_encode, batch_sequence
from torch import nn


def visualize_losses(running_train_loss, running_val_loss):
    plt.figure(figsize=(20, 10))

    plt.plot(range(len(running_train_loss)), running_train_loss, range(len(running_val_loss)), running_val_loss)

    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    plt.suptitle("Loss over iterations (LSTM)", fontsize=20)
    plt.legend()

    plt.show()
    plt.close("all")

    return


def train_model(
    model,
    optimizer,
    train_data,
    test_data,
    criterion,
    epochs,
    batch_size,
    seq_length,
    max_norm,
    device=None,
    code_size=83,
    plot_losses=False
):

    running_train_loss = list()
    running_val_loss = list()

    num_batches = train_data.size // (seq_length*batch_size)

    ### Outer training loop
    for epoch in range(1, epochs + 1):
        train_h = model.init_hidden_state(mean=0.0, stddev=0.5)
        iteration = 0
        train_losses = list()

        ### Inner training loop
        for X, y in batch_sequence(train_data, batch_size, seq_length)[0]:
            X = one_hot_encode(X, code_size)
            X, y = torch.as_tensor(X).to(device), torch.as_tensor(y).to(device)

            model.train()
            iteration += 1

            train_h = (
                tuple([each.data.to(device) for each in train_h])
                if type(train_h) == tuple
                else train_h.data
            )
            optimizer.zero_grad()

            outputs, train_h = model(X, train_h)

            loss = criterion(
                outputs,
                y.reshape(
                    -1,
                ).long(),
            )

            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            train_losses.append(loss.item())

            ### Outer validation loop
            if (not iteration % 20) or (iteration == num_batches):
                i = 0
                val_losses = list()
                model.eval()
                val_h = model.init_hidden_state(mean=0.0, stddev=0.5)

                with torch.inference_mode():
                    ### Inner validation loop
                    for X_, y_ in batch_sequence(test_data, batch_size, seq_length)[0]:
                        i += 1

                        val_h = (
                            tuple([each.data.to(device) for each in val_h])
                            if type(val_h) == tuple
                            else val_h.data
                        )

                        X_ = torch.as_tensor(one_hot_encode(X_, code_size)).to(device)
                        y_ = torch.as_tensor(y_).to(device)

                        outputs_, val_h = model(X_, val_h)

                        val_loss = criterion(
                            outputs_,
                            y_.reshape(
                                -1,
                            ).long(),
                        )
                        val_losses.append(val_loss.item())

                    ### Report training and validation losses
                    val_loss = torch.Tensor(val_losses).mean().item()

                    train_loss = torch.Tensor(train_losses).mean().item()

                running_train_loss.append(train_loss)
                running_val_loss.append(val_loss)

                print("=" * 80)
                print(
                    f"Epoch: {epoch}/{epochs}, Iteration {iteration}/{num_batches},",
                    f"Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}",
                )

        print("\n" + "=" * 80)
        print("=" * 80, end="\n\n")
        # print('='*60)
        # print(f'Epoch: {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}\n')
        # print('='*60)

    if plot_losses:
        visualize_losses(running_train_loss, running_val_loss)

    return model, optimizer
