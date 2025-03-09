"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
Modified by Daeho Um (daehoum1@snu.ac.kr)
"""
import torch


def train_node(model, x, data, optimizer, critereon, train_loader=None, device="cuda"):
    model.train()

    return (
        train_sampled_node(model, train_loader, x, data, optimizer, critereon, device)
        if train_loader
        else train_full_batch_node(model, x, data, optimizer, critereon)
    )


def train_full_batch_node(model, x, data, optimizer, critereon):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x, data.edge_index)[data.train_mask]
    y_true = data.y[data.train_mask].squeeze()

    loss = critereon(y_pred, y_true)
    loss.backward()
    optimizer.step()

    return loss


def train_sampled_node(model, train_loader, x, data, optimizer, critereon, device):
    model.train()

    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        x_batch = x[n_id]

        optimizer.zero_grad()
        y_pred = model(x_batch, adjs=adjs, full_batch=False)
        y_true = data.y[n_id[:batch_size]].squeeze()
        loss = critereon(y_pred, y_true)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        logger.debug(f"Batch loss: {loss.item():.2f}")

    return total_loss / len(train_loader)


@torch.no_grad()
def test_node(model, x, data, logits=None, evaluator=None, inference_loader=None, device="cuda"):
    model.eval()
    logits = inference_full_batch_node(model, x, data.edge_index)
    accs = []
    for _, mask in data("train_mask", "val_mask", "test_mask"):
        pred = logits[mask].max(1)[1]
        if evaluator:
            acc = evaluator.eval({"y_true": data.y[mask], "y_pred": pred.unsqueeze(1)})["acc"]
        else:
            acc = pred.eq(data.y[mask].squeeze()).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs, logits


def inference_full_batch_node(model, x, edge_index):
    out = model(x, edge_index)
    return out


def inference_sampled_node(model, x, inference_loader, device):
    return model.inference(x, inference_loader, device)
