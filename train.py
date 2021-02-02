import torch

from tqdm import tqdm

from utils import plotter


def Dimension_Reduction(train_dataloader, val_dataloader, model,
                        optimizer, loss_fn, n_epochs):
    global loss
    train_is_triplet = train_dataloader.dataset.is_triplet
    val_is_triplet = val_dataloader.dataset.is_triplet
    train_len = len(train_dataloader)
    val_len = len(val_dataloader)

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        val_loss = 0
        train_acc = 0
        val_acc = 0
        history = []

        for inputs, labels in tqdm(train_dataloader, desc="Train Iteration"):
            if train_is_triplet:
                anchor, pos, neg = inputs
                anchor = anchor.to(device='cuda')
                pos = pos.to(device='cuda')
                neg = neg.to(device='cuda')
                outputs = model(anchor, pos, neg)
                loss = loss_fn(*outputs)

            elif not train_is_triplet:
                w0, w1 = inputs
                w0 = w0.to(device='cuda')
                w1 = w1.to(device='cuda')
                labels = labels.to(device='cuda')
                outputs = model(w0, w1)
                loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(val_dataloader, desc="Valu Iteration"):
                if val_is_triplet:
                    anchor, pos, neg = inputs
                    anchor = anchor.to(device='cuda')
                    pos = pos.to(device='cuda')
                    neg = neg.to(device='cuda')
                    outputs = model(anchor, pos, neg)
                    loss = loss_fn(*outputs)
                elif not val_is_triplet:
                    w0, w1 = inputs
                    w0 = w0.to(device='cuda')
                    w1 = w1.to(device='cuda')
                    labels = labels.to(device='cuda')
                    outputs = model(w0, w1)
                    loss = loss_fn(outputs, labels)

                val_loss += loss.item()

        epoch_train_loss = train_loss / train_len
        epoch_val_loss = val_loss / val_len

        history.append([epoch, epoch_train_loss, epoch_val_loss])

        plotter(history)

        return model.state_dict()


def fit(train_dataloader, val_dataloader, model, optimizer, loss_fn, n_epochs, post_classification=False,
        with_acc=False, with_scheduler=False):
    train_is_triplet = train_dataloader.dataset.is_triplet
    val_is_triplet = val_dataloader.dataset.is_triplet
    train_len = len(train_dataloader)
    val_len = len(val_dataloader)
    history = []
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        val_loss = 0
        train_acc = 0
        val_acc = 0

        for inputs, labels in tqdm(train_dataloader, desc="Train iteration"):
            # if with_scheduler:
            #     prev_sd = model.state_dict()
            #
            # optimizer.zero_grad()
            if train_is_triplet:
                anchor, pos, neg = inputs
                anchor = anchor.to(device='cuda')
                pos = pos.to(device='cuda')
                neg = neg.to(device='cuda')
                outputs = model(anchor, pos, neg)
                if not post_classification:
                    loss = loss_fn(*outputs)
                elif post_classification:
                    y0 = torch.tensor([0 for _ in range(len(outputs[0]))]).to(device='cuda')
                    y1 = torch.tensor([1 for _ in range(len(outputs[1]))]).to(device='cuda')
                    y = torch.cat((y0, y1))
                    outputs = torch.cat((outputs[0], outputs[1]))
                    loss = loss_fn(outputs, y)

            elif not train_is_triplet:
                w0, w1 = inputs
                w0 = w0.to(device='cuda')
                w1 = w1.to(device='cuda')
                labels = labels.to(device='cuda')

                outputs = model(w0, w1)
                loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            if with_acc:
                if post_classification:
                    train_acc += accuracy_score(outputs, y)
                elif not post_classification:
                    train_acc += accuracy_score(outputs, labels)

            train_loss += loss.item()

        model.eval();
        with torch.no_grad():
            for inputs, labels in tqdm(val_dataloader, desc="Val iteration"):
                if val_is_triplet:
                    anchor, pos, neg = inputs
                    anchor = anchor.to(device='cuda')
                    pos = pos.to(device='cuda')
                    neg = neg.to(device='cuda')
                    outputs = model(anchor, pos, neg)
                    loss = loss_fn(*outputs)
                else:
                    w0, w1 = inputs
                    w0 = w0.to(device='cuda')
                    w1 = w1.to(device='cuda')
                    labels = labels.to(device='cuda')
                    if not post_classification:
                        outputs = model(w0, w1)
                    elif post_classification:
                        outputs = model.classifire_it(w0, w1)
                    loss = loss_fn(outputs, labels)

                val_loss += loss.item()
                if with_acc:
                    val_acc += accuracy_score(outputs, labels)

        epoch_train_loss = train_loss / train_len
        epoch_val_loss = val_loss / val_len

        if with_acc:
            epoch_train_acc = train_acc / train_len
            epoch_val_acc = val_acc / val_len

        if with_scheduler:
            flag = False
            if epoch > 2:
                flag = True
                if epoch_val_loss > prev_loss:
                    model.load_state_dict(prev_sd)
                    epoch_val_loss = prev_loss
                    optimizer.param_groups[0]['lr'] /= 1.5
                    flag = False
            if (epoch <= 2) or (flag == True):
                history.append([epoch, epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc])
                flag = False
            prev_loss = epoch_val_loss

        elif not with_scheduler:
            if with_acc:
                history.append([epoch, epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc])
            else:
                history.append([epoch, epoch_train_loss, epoch_val_loss])

        plotter(history)

    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    return model.state_dict()

