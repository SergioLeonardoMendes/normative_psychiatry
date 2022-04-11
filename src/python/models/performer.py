"""
Performer implementation.
Wrapper for the Performer implementation from https://github.com/lucidrains/performer-pytorch
"""
from performer_pytorch import PerformerLM
from transformer import TransformerBase


class Performer(TransformerBase):
    def __init__(
            self,
            ordering,
            vocab_size: int,
            max_seq_len: int,
            n_embd: int,
            n_layer: int,
            n_heads: int = 8,
            local_attn_heads: int = 0,
            local_window_size: int = 256,
            ff_mult: int = 4,
            ff_glu: bool = True,
            rotary_position_emb: bool = True,
            axial_position_emb: bool = False,
            emb_dropout: float = 0.0,
            ff_dropout: float = 0.0,
            attn_dropout: float = 0.0,
    ) -> None:
        """
        Performer implementation wrapped with the ordering object that transforms 2D/3D images into 1D sequences.

        Args:
            ordering: ordering object used to flat the input
            vocab_size: number of different tokens that the transformer support
            max_seq_len: max sequence length
            n_embd: embedding dimension
            n_layer: number of layers
            n_head: number of heads in the self attention mechanism
            emb_dropout: drop probability for the Dropout layer just after the embedding layer
            ff_dropout: drop probability for the Dropout layer just after the feedforward layers
            attn_dropout: drop probability for the Dropout layer just after the attention mechanism
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_heads = n_heads
        self.local_attn_heads = local_attn_heads,
        self.local_window_size = local_window_size,
        self.ff_mult = ff_mult,
        self.ff_glu = ff_glu
        self.rotary_position_emb = rotary_position_emb,
        self.axial_position_emb = axial_position_emb,
        self.emb_dropout = emb_dropout
        self.ff_dropout = ff_dropout
        self.attn_dropout = attn_dropout

        self.model = PerformerLM(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            dim=n_embd,
            depth=n_layer,
            heads=n_heads,
            local_attn_heads=local_attn_heads,
            local_window_size=local_window_size,
            ff_mult=ff_mult,
            ff_glu=ff_glu,
            rotary_position_emb=rotary_position_emb,
            axial_position_emb=axial_position_emb,
            emb_dropout=emb_dropout,
            ff_dropout=ff_dropout,
            attn_dropout=attn_dropout,
            causal=True,
            auto_check_redraw=False,
        )

        self.ordering = ordering

    def get_ordering(self):
        return self.ordering
