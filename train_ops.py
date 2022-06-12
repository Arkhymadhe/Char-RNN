











def train_model(model, opt, train_data, test_data, criterion,
                epochs, train_data, batch_size, seq_length,
                max_norm, device = None, code_size = 83):
    ### Outer training loop
    for epoch in range(1, epochs + 1):
        h = model.init_hidden_state(mean = 0., stddev = .5)
        iteration = 0
        train_losses = list()

        ### Inner training loop
        for X, y in batch_sequence(train_data, batch_size, seq_length)[0]:
            X = one_hot_encode(X, code_size)
            X, y = torch.as_tensor(X).to(device), torch.as_tensor(y).to(device)

            model.train()
            iteration += 1

            h = tuple([each.data.to(device) for each in h])
            opt.zero_grad()

            outputs, h = model(X, h)

            loss = criterion(outputs, y.reshape(-1,).long())

            loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            opt.step()

            train_losses.append(loss.item())

            ### Outer validation loop
            if (not iteration % 20) or (iteration == num_batches):
                i = 0
                val_losses = list()
                model.eval()
                h_ = model.init_hidden_state(mean = 0., stddev = .5)

                ### Inner validation loop
                for X_, y_ in batch_sequence(test_data, batch_size, seq_length)[0]:
                    i += 1

                    h_ = tuple([each.data.to(device) for each in h_])

                    X_ = torch.as_tensor(one_hot_encode(X_, code_size)).to(device)
                    y_ = torch.as_tensor(y_).to(device)

                    outputs_, h_ = model(X_, h_)

                    val_loss = criterion(outputs_, y_.reshape(-1,).long())
                    val_losses.append(val_loss.item())

                ### Report training and validation losses
                val_loss = torch.Tensor(val_losses).mean().item()

                train_loss = torch.Tensor(train_losses).mean().item()

                print('='*80)
                print(f'Epoch: {epoch}/{epochs}, Iteration {iteration}/{num_batches},',
                      f'Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}')

        print('\n'+'='*80)
        print('='*80)
        #print('='*60)
        #print(f'Epoch: {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}\n')
        #print('='*60)

