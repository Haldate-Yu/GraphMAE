import time
import copy
import logging
import torch
import torch.nn as nn
from torch_geometric.data import NeighborSampler
from tqdm import tqdm

from graphmae.utils import create_optimizer, accuracy, get_missing_feature_mask
from missing_feature_process.data_utils import set_train_val_test_split
from missing_feature_process.models import get_model
from missing_feature_process.train_node import train_node
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from missing_feature_process.train_node import test_node


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, x, edge_index, *args):
        logits = self.linear(x)
        return logits


def node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device,
                                   linear_prob=True, mute=False):
    model.eval()
    if linear_prob:
        with torch.no_grad():
            x = model.embed(x.to(device), graph.edge_index.to(device))
            in_feat = x.shape[1]
        encoder = LogisticRegression(in_feat, num_classes)
    else:
        encoder = model.encoder
        encoder.reset_classifier(num_classes)

    num_finetune_params = [p.numel() for p in encoder.parameters() if p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")

    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    final_acc, estp_acc = linear_probing_for_transductive_node_classiifcation(encoder, graph, x, optimizer_f,
                                                                              max_epoch_f, device, mute)
    return final_acc, estp_acc


def linear_probing_for_transductive_node_classiifcation(model, graph, feat, optimizer, max_epoch, device, mute=False):
    criterion = torch.nn.CrossEntropyLoss()

    graph = graph.to(device)
    x = feat.to(device)

    train_mask = graph.train_mask
    val_mask = graph.val_mask
    test_mask = graph.test_mask
    labels = graph.y

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        # only encoder part
        out = model(x, graph.edge_index)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            # use original feature to val/test
            pred = model(x, graph.edge_index)
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(
                f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

    best_model.eval()
    with torch.no_grad():
        # use original feature to val/test
        pred = model(x, graph.edge_index)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
    if mute:
        print(
            f"\n# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
    else:
        print(
            f"\n--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    # (final_acc, es_acc, best_acc)
    return test_acc, estp_test_acc


def missing_feature_node_classification_evaluation(args, seed, model, graph, split_idx, x, num_classes,
                                                   lr_f, weight_decay_f,
                                                   max_epoch_f, device,
                                                   mute=False):
    model.eval()
    num_finetune_params = [p.numel() for p in model.parameters() if p.requires_grad]
    if not mute:
        print(f"num parameters for finetuneing: {sum(num_finetune_params)}")

    split_idx = split_idx
    n_nodes, n_features = graph.x.shape
    num_classes = num_classes
    train_loader = (
        NeighborSampler(
            graph.edge_index,
            node_idx=split_idx["train"],
            sizes=[15, 10, 5][: args.downstream_num_layers],
            batch_size=args.downstream_batch_size,
            shuffle=True,
            num_workers=12,
        )
        if args.downstream_graph_sampling
        else None
    )
    inference_loader = (
        NeighborSampler(
            graph.edge_index, node_idx=None, sizes=[-1], batch_size=4096, shuffle=False, num_workers=12,
        )
        if args.downstream_graph_sampling
        else None
    )

    data = (set_train_val_test_split(
        seed=seed, data=graph, split_idx=split_idx, dataset_name=args.dataset, )
            .to(device))
    if args.dataset in ["ogbn_arxiv", "ogbn-products"]:
        evaluator = Evaluator(name=args.dataset)
    else:
        evaluator = None

    missing_feature_mask = (get_missing_feature_mask(
        rate=args.mask_rate, n_nodes=n_nodes, n_features=n_features, type=args.feature_mask_type, )
                            .to(device))
    x = data.x.clone()
    if args.feature_init_type == "zero":
        x[~missing_feature_mask] = float(0)
    elif args.feature_init_type == "random":
        init_x = torch.randn_like(x)
        x[~missing_feature_mask] = init_x[~missing_feature_mask]
    else:
        raise ValueError(f"{args.feature_init_type} not implemented!")

    if args.downstream_model in ["gcnmf", "pagnn"]:
        filled_features = torch.full_like(x, float("nan"))
    else:
        # use GraphMAE to fill missing features
        if args.feature_mask_type == "structural":
            mask_node_ids = torch.where(missing_feature_mask.sum(dim=1) == 0)[0]
        elif args.feature_mask_type == "uniform":
            mask_node_ids = torch.where(missing_feature_mask.sum(dim=1) != missing_feature_mask.shape[1])[0]
        else:
            raise ValueError(f"{args.feature_mask_type} not implemented!")

        node_mask = torch.ones(x.shape[0], dtype=torch.bool)
        node_mask[mask_node_ids] = False
        filled_features = model.missing_attr_prediction(x, data.edge_index, node_mask, args.feature_mask_type).detach()
        # set a threshold?
        filled_features[filled_features < 0] = 0

    downstream_model = get_model(
        model_name=args.downstream_model,
        num_features=data.num_features,
        num_classes=num_classes,
        edge_index=data.edge_index,
        x=x,
        mask=missing_feature_mask,
        args=args,
    ).to(device)
    downstream_params = list(downstream_model.parameters())

    optimizer = torch.optim.Adam(downstream_params, lr=lr_f, weight_decay=weight_decay_f)
    criterion = torch.nn.NLLLoss()

    epoch_test_acc = 0
    best_val_acc = 0
    best_model = None
    for epoch in range(0, max_epoch_f):
        x = torch.where(missing_feature_mask, data.x, filled_features)
        train_node(
            downstream_model, x, data, optimizer, criterion, train_loader=train_loader, device=device,
        )
        (train_acc, val_acc, epoch_test_acc), out = test_node(
            downstream_model, x=x, data=data, evaluator=evaluator, inference_loader=inference_loader, device=device,
        )
        if epoch == 0 or val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(downstream_model)
        if epoch > args.patience:
            break
        if not mute:
            print(
                f"Epoch {epoch + 1} - Train acc: {train_acc:.3f}, Val acc: {val_acc:.3f}, Test acc: {epoch_test_acc:.3f}"
            )
    best_model.eval()
    (train_acc, val_acc, tmp_test_acc), out = test_node(
        best_model, x=x, data=data, evaluator=evaluator, inference_loader=inference_loader, device=device,
    )
    if not mute:
        print(f"Final Test acc: {tmp_test_acc:.3f}")
    return epoch_test_acc, tmp_test_acc
