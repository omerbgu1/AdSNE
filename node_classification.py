import torch
from tqdm import tqdm
import torch.nn.functional as F


def train_node_classifier(model, data, optimizer, epochs, disable_tqdm=False):
    model.train()
    for _ in tqdm(range(epochs), desc='node classification train', disable=disable_tqdm):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        # ce
        # loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        # ce with logits
        num_classes = out.shape[1]
        targets = F.one_hot(data.y[data.train_mask], num_classes=num_classes).float()

        loss = F.binary_cross_entropy_with_logits(out[data.train_mask], targets)
        loss.backward()
        optimizer.step()
    return


@torch.no_grad()
def test_node_classifier(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    preds = out.argmax(dim=1)
    correct = preds[data.test_mask] == data.y[data.test_mask]
    accuracy = int(correct.sum()) / int(data.test_mask.sum())
    return accuracy
