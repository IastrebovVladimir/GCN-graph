import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from graph_utils import load_graph_from_edge_list


def load_features(path: str, n: int) -> torch.Tensor:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = list(map(float, line.split()))
            data.append(row)

    x = torch.tensor(data, dtype=torch.float32)

    if x.size(0) != n:
        raise ValueError(f"В features.txt {x.size(0)} строк, а вершин в графе {n}")

    return x


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    n = adj.size(0)
    device = adj.device

    adj_tilde = adj + torch.eye(n, device=device)
    degree = adj_tilde.sum(dim=1)

    deg_inv_sqrt = degree.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj_tilde @ D_inv_sqrt
    return adj_norm


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        x = adj_norm @ x
        x = self.linear(x)
        return x


class GCN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int,
                 out_features: int, dropout: float = 0.3):
        super().__init__()
        self.gcn1 = GCNLayer(in_features, hidden_features)
        self.gcn2 = GCNLayer(hidden_features, out_features)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        x = self.gcn1(x, adj_norm)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, adj_norm)
        return x


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def main():
    torch.manual_seed(42)

    adj = load_graph_from_edge_list("graph.txt", zero_based=True)
    n = adj.size(0)

    if os.path.exists("features.txt"):
        x = load_features("features.txt", n)
        print(f"Загружены признаки из features.txt: shape = {tuple(x.shape)}")
    else:
        x = torch.randn(n, 8)
        print(f"features.txt не найден, использую случайные признаки: shape = {tuple(x.shape)}")

    labels = torch.zeros(n, dtype=torch.long)
    labels[20:] = 1

    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[0:10] = True
    train_mask[20:30] = True
    test_mask = ~train_mask

    adj_norm = normalize_adjacency(adj)

    model = GCN(
        in_features=x.size(1),
        hidden_features=32,
        out_features=2,
        dropout=0.3
    )

    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    epochs = 300
    history = {"epoch": [], "loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(x, adj_norm)
        loss = criterion(logits[train_mask], labels[train_mask])

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_eval = model(x, adj_norm)
            train_acc = accuracy(logits_eval[train_mask], labels[train_mask])
            test_acc = accuracy(logits_eval[test_mask], labels[test_mask])

        history["epoch"].append(epoch)
        history["loss"].append(loss.item())
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss.item():.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Test Acc: {test_acc:.4f}"
            )

    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["loss"], label="Train loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GCN: динамика функции потерь")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["train_acc"], label="Train acc", color="blue")
    plt.plot(history["epoch"], history["test_acc"], label="Test acc", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("GCN: точность на train/test")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy.png", dpi=200)
    plt.close()

    print("Сохранены картинки: loss.png и accuracy.png")

    model.eval()
    with torch.no_grad():
        logits = model(x, adj_norm)
        preds = logits.argmax(dim=1)

    print("Финальные предсказания по вершинам:")
    for i in range(n):
        print(f"Вершина {i:02d}: предсказанный класс {preds[i].item()} (истинный: {labels[i].item()})")


if __name__ == "__main__":
    main()
