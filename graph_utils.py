import torch


def load_graph_from_edge_list(path: str, zero_based: bool = True) -> torch.Tensor:
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().strip().split()
        n, m = map(int, first)

        adj = torch.zeros((n, n), dtype=torch.float32)
        edge_count = 0

        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v = map(int, line.split())
            if not zero_based:
                u -= 1
                v -= 1
            adj[u, v] = 1.0
            adj[v, u] = 1.0
            edge_count += 1

    if edge_count != m:
        print(f"Предупреждение: в заголовке M={m}, а реально прочитано {edge_count} рёбер")

    return adj
