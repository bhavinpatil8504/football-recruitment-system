"""
Stage 3: Representation Learning — Graph Neural Network
---------------------------------------------------------
Models the passing network of each match as a graph:
  - Nodes  = players (with node features from Stage 2)
  - Edges  = passes between players (weighted by frequency)

Supports two architectures (selectable via --gnn-arch flag):
  - GAT  (Veličković et al., 2018): 4-head attention, learns which
    passing connections matter most. Default choice.
  - GraphSAGE (Hamilton et al., 2017): neighborhood aggregation,
    scales better but no attention interpretability.

Training objective: self-supervised link prediction (predict whether
two players exchange passes). No labeled data required.

Embeddings capture a player's RELATIONAL role within the tactical
system — who they pass to, who passes to them, and how that
compares to other players in similar positions.

Install: pip install torch torch-geometric
(CPU-only torch is fine for this scale of data.)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ── Graph construction ─────────────────────────────────────────────────────────
def build_passing_graph(match_events: pd.DataFrame,
                        player_features: pd.DataFrame,
                        feature_cols: list) -> Data:
    """
    Build a PyG Data object from a single match's events.

    Parameters
    ----------
    match_events    : events DataFrame filtered to one match
    player_features : wide feature DataFrame (from Stage 2)
    feature_cols    : which columns to use as node features

    Returns
    -------
    torch_geometric.data.Data or None if graph has < 2 nodes
    """
    # ── Filter to pass events ─────────────────────────────────────────────────
    def type_name(t):
        if isinstance(t, dict):
            return t.get("name", "")
        return str(t)

    passes = match_events[match_events["type"].apply(type_name) == "Pass"].copy()
    passes = passes.dropna(subset=["player_id"])

    # ── Unique players in this match ──────────────────────────────────────────
    all_player_ids = (
        list(passes["player_id"].dropna()) +
        list(passes["pass_recipient_id"].dropna()
             if "pass_recipient_id" in passes.columns else [])
    )
    players_in_match = pd.Series(all_player_ids).unique()

    if len(players_in_match) < 2:
        return None

    # Map player_id → node index
    player_to_node = {pid: i for i, pid in enumerate(players_in_match)}
    n_nodes        = len(players_in_match)

    # ── Node features ─────────────────────────────────────────────────────────
    season_id      = match_events["season_id"].iloc[0] if "season_id" in match_events.columns else None
    competition_id = match_events["competition_id"].iloc[0] if "competition_id" in match_events.columns else None

    node_feats = []
    for pid in players_in_match:
        mask = (player_features["player_id"] == pid)
        if season_id is not None:
            mask &= (player_features["season_id"] == season_id)
        if competition_id is not None:
            mask &= (player_features["competition_id"] == competition_id)

        row = player_features[mask]
        if len(row) > 0:
            vals = row[feature_cols].values[0].astype(float)
        else:
            vals = np.zeros(len(feature_cols))
        node_feats.append(vals)

    x = torch.tensor(np.array(node_feats), dtype=torch.float)

    # ── Edge construction ─────────────────────────────────────────────────────
    # Edge weight = number of passes between the pair (directed)
    edge_counts = {}
    for _, row in passes.iterrows():
        src_id = row.get("player_id")
        tgt_id = row.get("pass_recipient_id") if "pass_recipient_id" in row.index else None
        if src_id in player_to_node and tgt_id in player_to_node:
            src = player_to_node[src_id]
            tgt = player_to_node[tgt_id]
            edge_counts[(src, tgt)] = edge_counts.get((src, tgt), 0) + 1

    if not edge_counts:
        return None

    edge_index = torch.tensor(list(zip(*edge_counts.keys())), dtype=torch.long)
    edge_weight = torch.tensor(list(edge_counts.values()), dtype=torch.float)
    # Normalize edge weights
    edge_weight = edge_weight / edge_weight.max()

    # ── Node labels (player_ids for post-hoc lookup) ──────────────────────────
    player_ids = torch.tensor([int(pid) if isinstance(pid, (int, float)) else 0
                               for pid in players_in_match], dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight.unsqueeze(1),
        player_ids=player_ids,
        num_nodes=n_nodes,
    )


def build_all_match_graphs(events: pd.DataFrame,
                            player_features: pd.DataFrame,
                            feature_cols: list) -> list:
    """Build a graph for every match in the events DataFrame."""
    graphs = []
    match_ids = events["match_id"].unique()
    print(f"  Building graphs for {len(match_ids)} matches...")

    for mid in tqdm(match_ids, desc="Building graphs"):
        match_ev = events[events["match_id"] == mid]
        g = build_passing_graph(match_ev, player_features, feature_cols)
        if g is not None:
            g.match_id = int(mid)
            graphs.append(g)

    print(f"  Built {len(graphs)} valid graphs (from {len(match_ids)} matches)")
    return graphs


# ── GraphSAGE model ───────────────────────────────────────────────────────────
class PassingNetworkGNN(nn.Module):
    """
    GraphSAGE encoder for passing networks.

    Architecture:
      Input features  → SAGEConv(64) → ReLU → Dropout
                      → SAGEConv(32) → ReLU
                      → output: 32-dim embedding per node

    The embedding captures a player's ROLE within the tactical system,
    not just their individual stats.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 64,
                 embed_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.conv1   = SAGEConv(in_channels, hidden_dim)
        self.conv2   = SAGEConv(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Auxiliary head: reconstruct edge existence (self-supervised)
        self.edge_head = nn.Linear(embed_dim * 2, 1)

    def encode(self, x: torch.Tensor,
               edge_index: torch.Tensor) -> torch.Tensor:
        """Return per-node embeddings."""
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        return h  # shape: (n_nodes, embed_dim)

    def forward(self, x, edge_index, edge_label_index=None):
        z = self.encode(x, edge_index)
        if edge_label_index is not None:
            src = z[edge_label_index[0]]
            tgt = z[edge_label_index[1]]
            return z, self.edge_head(torch.cat([src, tgt], dim=-1)).squeeze()
        return z, None


# ── GAT model (Graph Attention Network) ──────────────────────────────────────
class PassingNetworkGAT(nn.Module):
    """
    Graph Attention Network encoder for passing networks.

    Key advantage over GraphSAGE: ATTENTION WEIGHTS tell us which
    passing connections matter most for each player's role.
    This is more interpretable and typically learns richer representations.

    Architecture:
      Input features  → GATConv(64, heads=4) → ELU → Dropout
                      → GATConv(32, heads=1) → output: 32-dim embedding

    Reference: Veličković et al. (2018), "Graph Attention Networks"
    """

    def __init__(self, in_channels: int, hidden_dim: int = 64,
                 embed_dim: int = 32, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.conv1   = GATConv(in_channels, hidden_dim // heads,
                               heads=heads, dropout=dropout)
        self.conv2   = GATConv(hidden_dim, embed_dim,
                               heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        # Auxiliary head: reconstruct edge existence (self-supervised)
        self.edge_head = nn.Linear(embed_dim * 2, 1)

    def encode(self, x: torch.Tensor,
               edge_index: torch.Tensor) -> torch.Tensor:
        """Return per-node embeddings."""
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        return h  # shape: (n_nodes, embed_dim)

    def forward(self, x, edge_index, edge_label_index=None):
        z = self.encode(x, edge_index)
        if edge_label_index is not None:
            src = z[edge_label_index[0]]
            tgt = z[edge_label_index[1]]
            return z, self.edge_head(torch.cat([src, tgt], dim=-1)).squeeze()
        return z, None


# ── Self-supervised training (link prediction) ────────────────────────────────
def train_gnn(graphs: list,
              in_channels: int,
              hidden_dim: int = 64,
              embed_dim: int = 32,
              epochs: int = 30,
              lr: float = 1e-3,
              device: str = "cpu",
              architecture: str = "gat") -> nn.Module:
    """
    Train the GNN using link prediction as a self-supervised objective.

    For each graph:
      - Positive edges  = actual passes
      - Negative edges  = random non-edges (same count)

    No labelled data required. The model learns to encode
    which players interact, capturing tactical role structure.

    Parameters
    ----------
    architecture : 'sage' or 'gat' (default: 'gat')
        GAT adds attention weights for interpretability.
    """
    if architecture == "gat":
        model = PassingNetworkGAT(in_channels, hidden_dim, embed_dim).to(device)
        print(f"  Architecture: GAT (Graph Attention Network)")
    else:
        model = PassingNetworkGNN(in_channels, hidden_dim, embed_dim).to(device)
        print(f"  Architecture: GraphSAGE")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    print(f"  Training GNN: {epochs} epochs, device={device}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches  = 0

        for g in graphs:
            g = g.to(device)
            if g.edge_index.shape[1] < 2:
                continue

            # Positive edges (actual passes)
            pos_edge = g.edge_index                       # (2, E)
            n_pos    = pos_edge.shape[1]

            # Negative edges (random non-edges)
            neg_src  = torch.randint(0, g.num_nodes, (n_pos,), device=device)
            neg_tgt  = torch.randint(0, g.num_nodes, (n_pos,), device=device)
            neg_edge = torch.stack([neg_src, neg_tgt], dim=0)

            edge_label_index = torch.cat([pos_edge, neg_edge], dim=1)
            labels = torch.cat([
                torch.ones(n_pos, device=device),
                torch.zeros(n_pos, device=device)
            ])

            optimizer.zero_grad()
            _, logits = model(g.x, g.edge_index, edge_label_index)
            if logits is None:
                continue

            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f}")

    return model


# ── Embedding extraction ───────────────────────────────────────────────────────
@torch.no_grad()
def extract_player_embeddings(model: PassingNetworkGNN,
                               graphs: list,
                               device: str = "cpu") -> pd.DataFrame:
    """
    Run all graphs through the trained GNN and collect per-player embeddings.
    
    A player appears in multiple matches → we average their embeddings 
    across all matches. This gives a stable, season-long representation.

    Returns
    -------
    pd.DataFrame with columns: player_id, emb_0, emb_1, ..., emb_31
    """
    model.eval()
    player_embeds = {}   # player_id → list of embedding vectors

    for g in tqdm(graphs, desc="Extracting embeddings"):
        g   = g.to(device)
        z, _ = model(g.x, g.edge_index)
        z   = z.cpu().numpy()   # (n_nodes, embed_dim)

        for i, pid in enumerate(g.player_ids.tolist()):
            if pid == 0:
                continue
            if pid not in player_embeds:
                player_embeds[pid] = []
            player_embeds[pid].append(z[i])

    # Average across all match appearances
    records = []
    for pid, embeds in player_embeds.items():
        avg_embed = np.mean(embeds, axis=0)
        row = {"player_id": pid}
        for j, val in enumerate(avg_embed):
            row[f"emb_{j}"] = val
        records.append(row)

    df = pd.DataFrame(records)
    print(f"  Embeddings computed for {len(df)} players, "
          f"dim={len([c for c in df.columns if c.startswith('emb_')])}")
    return df


# ── Embedding visualization (t-SNE / UMAP) ──────────────────────────────────
def visualize_embeddings(embeddings: pd.DataFrame,
                         player_features: pd.DataFrame,
                         lineups: pd.DataFrame = None,
                         save_dir: str = "outputs",
                         method: str = "tsne") -> None:
    """
    Produce 2D scatter of GNN embeddings, colored by position.

    This visualization serves two purposes:
      1. Shows whether the GNN learns meaningful role clusters
         (if it does, positional groups should separate in 2D)
      2. Reveals sub-role structure — e.g., attacking midfielders
         clustering closer to forwards than defensive midfielders

    Parameters
    ----------
    method : 'tsne' or 'umap'
        t-SNE is the default (always available with sklearn).
        UMAP requires: pip install umap-learn
    """
    import os
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)

    emb_cols = [c for c in embeddings.columns if c.startswith("emb_")]
    if len(emb_cols) < 2:
        print("  Not enough embedding dimensions for visualization")
        return

    X = embeddings[emb_cols].values

    # Reduce to 2D
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
            X_2d = reducer.fit_transform(X)
        except ImportError:
            print("  umap-learn not installed, falling back to t-SNE")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(30, len(X) - 1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_2d = reducer.fit_transform(X)

    # Get positions for coloring
    from src.optimization.squad_optimizer import POSITION_MAP
    pos_series = None

    if lineups is not None and "position" in lineups.columns:
        # Use lineup-based position
        pos_map = (
            lineups[["player_id", "position"]]
            .drop_duplicates("player_id")
            .set_index("player_id")["position"]
        )
        pos_series = embeddings["player_id"].map(pos_map)
    elif "position" in player_features.columns:
        pf_map = (
            player_features[["player_id", "position"]]
            .drop_duplicates("player_id")
            .set_index("player_id")["position"]
        )
        pos_series = embeddings["player_id"].map(pf_map)

    if pos_series is None:
        pos_series = pd.Series(["unknown"] * len(embeddings))

    # Map to broad groups
    pos_groups = pos_series.apply(
        lambda p: POSITION_MAP.get(p, p) if isinstance(p, str) else "UNK"
    ).fillna("UNK")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {"GK": "#1f77b4", "DEF": "#2ca02c", "MID": "#ff7f0e",
              "ATT": "#d62728", "UNK": "#999999"}

    for pos, color in colors.items():
        mask = pos_groups == pos
        if mask.sum() > 0:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                      c=color, label=pos, alpha=0.6, s=20)

    ax.legend(title="Position", loc="best")
    ax.set_title(f"GNN Embedding Visualization ({method.upper()})\n"
                 f"Clusters = learned tactical roles from passing networks")
    ax.set_xlabel(f"{method.upper()} dimension 1")
    ax.set_ylabel(f"{method.upper()} dimension 2")
    plt.tight_layout()

    path = os.path.join(save_dir, f"gnn_embeddings_{method}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def build_position_baseline(player_features: pd.DataFrame,
                            lineups: pd.DataFrame) -> pd.DataFrame:
    """
    Create a one-hot position encoding baseline (no GNN needed).

    If the GNN ablation shows +GNN beats +Phase but NOT +Phase+OneHotPosition,
    then the GNN is just learning position — expensive and unnecessary.
    If +GNN beats +Phase+OneHotPosition, the GNN captures genuine relational
    structure beyond position identity.

    Returns DataFrame with columns: player_id, pos_GK, pos_DEF, pos_MID, pos_ATT
    """
    from src.optimization.squad_optimizer import assign_positions, POSITION_MAP

    df = player_features[["player_id"]].drop_duplicates().copy()

    # Get position for each player
    if "position" in player_features.columns:
        pos_map = (
            player_features[["player_id", "position"]]
            .drop_duplicates("player_id")
            .set_index("player_id")["position"]
        )
    elif lineups is not None:
        temp = assign_positions(player_features.copy(), lineups)
        pos_map = (
            temp[["player_id", "position"]]
            .drop_duplicates("player_id")
            .set_index("player_id")["position"]
        )
    else:
        pos_map = pd.Series("MID", index=df["player_id"])

    df["_pos"] = df["player_id"].map(pos_map).fillna("MID")

    for pos in ["GK", "DEF", "MID", "ATT"]:
        df[f"pos_{pos}"] = (df["_pos"] == pos).astype(float)

    return df.drop(columns=["_pos"])


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data from Stage 1 & 2
    events          = pd.read_parquet("outputs/raw_events.parquet")
    player_features = pd.read_parquet("outputs/player_features.parquet")

    # Select numerical feature columns for node features
    skip_cols = ["player_id", "player", "season_id", "competition_id"]
    feature_cols = [c for c in player_features.columns
                    if c not in skip_cols and
                    player_features[c].dtype in [np.float64, np.float32,
                                                   np.int64, np.int32]]

    # Normalise features before feeding to GNN
    scaler = StandardScaler()
    player_features[feature_cols] = scaler.fit_transform(
        player_features[feature_cols].fillna(0)
    )

    # Build graphs
    graphs = build_all_match_graphs(events, player_features, feature_cols)

    # Train GNN
    in_channels = len(feature_cols)
    model = train_gnn(graphs, in_channels=in_channels,
                      hidden_dim=64, embed_dim=32,
                      epochs=30, device=device)

    # Extract embeddings
    embeddings = extract_player_embeddings(model, graphs, device=device)

    embeddings.to_parquet("outputs/player_embeddings.parquet", index=False)
    torch.save(model.state_dict(), "outputs/gnn_model.pt")
    print("Saved embeddings and model weights to outputs/")
