import json
import logging
from pathlib import Path

import pytest
import torch

from config import Config
from data.organ_hierarchy import load_organ_hierarchy
from models.body_net import BodyNet
from models.hyperbolic.lorentz_loss import LorentzTreeRankingLoss
from train import load_precomputed_graph_distance_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "021201-19.yaml"
DATASET_INFO_PATH = PROJECT_ROOT / "Dataset" / "dataset_info.json"
TREE_PATH = PROJECT_ROOT / "Dataset" / "tree.json"
GRAPH_DISTANCE_PATH = PROJECT_ROOT / "Dataset" / "graph_distance_matrix.pt"

pytestmark = pytest.mark.skipif(
    not all(
        path.exists()
        for path in (
            CONFIG_PATH,
            DATASET_INFO_PATH,
            TREE_PATH,
            GRAPH_DISTANCE_PATH,
        )
    ),
    reason="021201-19 local config/Dataset artifacts are not available",
)


def _load_cfg() -> Config:
    return Config.from_yaml(str(CONFIG_PATH))


def _load_class_depths(cfg: Config):
    with open(cfg.dataset_info_file, encoding="utf-8") as f:
        class_names = json.load(f)["class_names"]
    return load_organ_hierarchy(cfg.tree_file, class_names)


def test_02120119_declares_graph_random_runtime_path():
    cfg = _load_cfg()

    assert cfg.num_classes == 70
    assert cfg.hyp_direction_mode == "random"
    assert cfg.hyp_distance_mode == "graph"
    assert cfg.lr_scheduler == "cosine_multiphase"
    assert cfg.graph_distance_matrix == "Dataset/graph_distance_matrix.pt"


def test_02120119_bodynet_forward_and_graph_loss_smoke():
    cfg = _load_cfg()
    torch.manual_seed(0)

    model = BodyNet(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        base_channels=cfg.base_channels,
        growth_rate=cfg.growth_rate,
        dense_layers=cfg.dense_layers,
        bn_size=cfg.bn_size,
        embed_dim=cfg.hyp_embed_dim,
        curv=cfg.hyp_curv,
        class_depths=_load_class_depths(cfg),
        min_radius=cfg.hyp_min_radius,
        max_radius=cfg.hyp_max_radius,
        direction_mode=cfg.hyp_direction_mode,
    )
    model.eval()

    inputs = torch.randn(1, cfg.in_channels, 8, 8, 8)
    labels = torch.randint(0, cfg.num_classes, (1, 8, 8, 8), dtype=torch.long)

    with torch.no_grad():
        logits, voxel_emb, label_emb = model(inputs)

    assert logits.shape == (1, cfg.num_classes, 8, 8, 8)
    assert voxel_emb.shape == (1, cfg.hyp_embed_dim, 8, 8, 8)
    assert label_emb.shape == (cfg.num_classes, cfg.hyp_embed_dim)

    graph_dist_matrix = load_precomputed_graph_distance_matrix(
        cfg.graph_distance_matrix,
        logging.getLogger("test_02120119_runtime_path"),
    )
    assert tuple(graph_dist_matrix.shape) == (cfg.num_classes, cfg.num_classes)

    criterion = LorentzTreeRankingLoss(
        tree_dist_matrix=graph_dist_matrix,
        margin=cfg.hyp_margin,
        curv=cfg.hyp_curv,
        num_samples_per_class=2,
        num_negatives=2,
        t_start=cfg.hyp_t_start,
        t_end=cfg.hyp_t_end,
        warmup_epochs=cfg.hyp_warmup_epochs,
        curriculum_epochs=cfg.hyp_curriculum_epochs,
    )
    loss = criterion(voxel_emb, labels, label_emb)

    assert torch.isfinite(loss)
