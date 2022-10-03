











def train_model(model, opt, train_data, test_data, criterion,
                epochs, train_data, batch_size, seq_length,
                max_norm, device = None, code_size = 83):
    ### Outer training loop
    for epoch in range(1, epochs + 1):
        train_h = model.init_hidden_state(mean = 0., stddev = .5)
        iteration = 0
        train_losses = list()

        ### Inner training loop
        for X, y in batch_sequence(train_data, batch_size, seq_length)[0]:
            X = one_hot_encode(X, code_size)
            X, y = torch.as_tensor(X).to(device), torch.as_tensor(y).to(device)

            model.train()
            iteration += 1

            train_h = tuple([each.data.to(device) for each in train_h]) if type(train_h) == tuple else train_h.data
            opt.zero_grad()

            outputs, train_h = model(X, train_h)

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
                val_h = model.init_hidden_state(mean = 0., stddev = .5)

                ### Inner validation loop
                for X_, y_ in batch_sequence(test_data, batch_size, seq_length)[0]:
                    i += 1

                    val_h = tuple([each.data.to(device) for each in val_h]) if type(val_h) == tuple else val_h.data

                    X_ = torch.as_tensor(one_hot_encode(X_, code_size)).to(device)
                    y_ = torch.as_tensor(y_).to(device)

                    outputs_, val_h = model(X_, val_h)

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

