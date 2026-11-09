from typing import cast

from mint.models.gemma import Gemma, GemmaConfig
from mint.nn.blocks import GQAGluBlock
from mint.utils.parallel import ParallelContext, apply_tp


def test_tp_model_heads_and_weights() -> None:
    cfg = GemmaConfig(
        embed_dim=64,
        hidden_dim=128,
        seq_length=32,
        vocab_size=100,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
    )
    model = Gemma(cfg, gradient_checkpointing=False)
    pctx = ParallelContext(
        rank=0,
        world_size=2,
        dp_rank=0,
        dp_size=1,
        tp_rank=0,
        tp_size=2,
        ddp_level=0,
        sp=False,
    )

    block = cast(GQAGluBlock, model.blocks[0])
    orig_wq_shape = block.gqa.wq.weight.shape
    apply_tp(model, pctx)

    assert block.gqa.n_heads == 2
    assert block.gqa.n_kv_heads == 1
    assert block.gqa.wq.weight.shape[0] == orig_wq_shape[0] // 2
    assert block.gqa.wo.weight.shape[1] == orig_wq_shape[0] // 2
