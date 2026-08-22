import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In the embedding layers we multiply those weights by √d_model
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        if d_model % 2 != 0:
            raise ValueError("d_model must be even")

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to the even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to the odd positions
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        end_pos = start_pos + x.size(1)
        if start_pos < 0:
            raise ValueError("start_pos cannot be negative")
        if end_pos > self.seq_len:
            raise ValueError(f"Position {end_pos} exceeds the configured maximum of {self.seq_len}")
        pe = self.get_buffer("pe")
        x = x + pe[:, start_pos:end_pos, :]
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.features = features
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # multiplied
        self.bias = nn.Parameter(torch.zeros(features))  # added

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.layer_norm uses the backend's fused implementation instead of
        # launching separate mean, standard-deviation, subtract, and divide ops.
        return F.layer_norm(x, (self.features,), self.alpha, self.bias, self.eps)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 and B2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float, use_fused_attention: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        if d_model % h != 0:
            raise ValueError("d_model must be divisible by the number of attention heads")

        self.d_k = d_model // h
        self.use_fused_attention = use_fused_attention
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention_scores: torch.Tensor | None = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Change (batch, length, d_model) into (batch, heads, length, d_k)."""
        return x.view(x.shape[0], x.shape[1], self.h, self.d_k).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Change (batch, heads, length, d_k) back into (batch, length, d_model)."""
        return x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)

    def _apply_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.use_fused_attention:
            x = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            self.attention_scores = None
            return x

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        return x

    @staticmethod
    def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
        dropout: nn.Dropout | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = query.shape[-1]

        # (batch, h, query_len, d_k) @ (batch, h, d_k, key_len)
        # --> (batch, h, query_len, key_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, torch.finfo(attention_scores.dtype).min)
        # (batch, h, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self._split_heads(self.w_q(q))
        key = self._split_heads(self.w_k(k))
        value = self._split_heads(self.w_v(v))
        x = self._apply_attention(query, key, value, mask)

        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = self._combine_heads(x)

        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)

    def forward_cached(
        self,
        q: torch.Tensor,
        key_value: torch.Tensor,
        mask: torch.Tensor | None,
        cache: "AttentionKVCache | None",
        *,
        static_key_value: bool,
    ) -> tuple[torch.Tensor, "AttentionKVCache"]:
        """Attend one decoding step while reusing projected keys and values."""
        query = self._split_heads(self.w_q(q))

        if static_key_value and cache is not None:
            key = cache.key
            value = cache.value
        else:
            key = self._split_heads(self.w_k(key_value))
            value = self._split_heads(self.w_v(key_value))
            if cache is not None:
                key = torch.cat((cache.key, key), dim=2)
                value = torch.cat((cache.value, value), dim=2)

        updated_cache = AttentionKVCache(key=key, value=value)
        x = self._apply_attention(query, key, value, mask)
        return self.w_o(self._combine_heads(x)), updated_cache


@dataclass
class AttentionKVCache:
    """Projected attention keys and values with shape (batch, heads, length, d_k)."""

    key: torch.Tensor
    value: torch.Tensor


@dataclass
class DecoderLayerCache:
    """Self- and cross-attention state belonging to one decoder layer."""

    self_attention: AttentionKVCache | None = None
    cross_attention: AttentionKVCache | None = None


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))

    def add(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        """Add an already pre-normalized sublayer result to the residual stream."""
        return x + self.dropout(sublayer_output)


class EncoderBlock(nn.Module):
    def __init__(
        self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.self_attention_residual = ResidualConnection(features, dropout)
        self.feed_forward_residual = ResidualConnection(features, dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        x = self.self_attention_residual(
            x,
            lambda normalized_x: self.self_attention_block(normalized_x, normalized_x, normalized_x, src_mask),
        )
        x = self.feed_forward_residual(x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, features: int, layers: list[EncoderBlock]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        for layer in self.layers:
            if not isinstance(layer, EncoderBlock):
                raise TypeError("Encoder contains an unexpected layer type")
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.self_attention_residual = ResidualConnection(features, dropout)
        self.cross_attention_residual = ResidualConnection(features, dropout)
        self.feed_forward_residual = ResidualConnection(features, dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor | None,
        tgt_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.self_attention_residual(
            x,
            lambda normalized_x: self.self_attention_block(normalized_x, normalized_x, normalized_x, tgt_mask),
        )
        x = self.cross_attention_residual(
            x,
            lambda normalized_x: self.cross_attention_block(normalized_x, encoder_output, encoder_output, src_mask),
        )
        x = self.feed_forward_residual(x, self.feed_forward_block)
        return x

    def forward_step(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor | None,
        cache: DecoderLayerCache | None,
    ) -> tuple[torch.Tensor, DecoderLayerCache]:
        """Decode one token and return the updated inference-only KV cache."""
        cache = DecoderLayerCache() if cache is None else cache

        normalized_x = self.self_attention_residual.norm(x)
        self_attention_output, self_attention_cache = self.self_attention_block.forward_cached(
            normalized_x,
            normalized_x,
            None,
            cache.self_attention,
            static_key_value=False,
        )
        x = self.self_attention_residual.add(x, self_attention_output)

        normalized_x = self.cross_attention_residual.norm(x)
        cross_attention_output, cross_attention_cache = self.cross_attention_block.forward_cached(
            normalized_x,
            encoder_output,
            src_mask,
            cache.cross_attention,
            static_key_value=True,
        )
        x = self.cross_attention_residual.add(x, cross_attention_output)
        x = self.feed_forward_residual(x, self.feed_forward_block)
        return x, DecoderLayerCache(self_attention_cache, cross_attention_cache)


class Decoder(nn.Module):
    def __init__(self, features: int, layers: list[DecoderBlock]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = LayerNormalization(features)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor | None,
        tgt_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        for layer in self.layers:
            if not isinstance(layer, DecoderBlock):
                raise TypeError("Decoder contains an unexpected layer type")
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

    def forward_step(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor | None,
        cache: list[DecoderLayerCache] | None,
    ) -> tuple[torch.Tensor, list[DecoderLayerCache]]:
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} decoder cache entries, received {len(cache)}")

        updated_cache: list[DecoderLayerCache] = []
        for index, layer in enumerate(self.layers):
            if not isinstance(layer, DecoderBlock):
                raise TypeError("Decoder contains an unexpected layer type")
            layer_cache = None if cache is None else cache[index]
            x, new_layer_cache = layer.forward_step(x, encoder_output, src_mask, layer_cache)
            updated_cache.append(new_layer_cache)
        return self.norm(x), updated_cache


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
        return self.project(decoder_output)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(
        self,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor | None,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def decode_step(
        self,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor | None,
        tgt_token: torch.Tensor,
        position: int,
        cache: list[DecoderLayerCache] | None = None,
    ) -> tuple[torch.Tensor, list[DecoderLayerCache]]:
        """Decode one target position, reusing attention state from earlier positions."""
        if tgt_token.ndim != 2 or tgt_token.size(1) != 1:
            raise ValueError("tgt_token must have shape (batch, 1)")
        tgt = self.tgt_embed(tgt_token)
        tgt = self.tgt_pos(tgt, start_pos=position)
        return self.decoder.forward_step(tgt, encoder_output, src_mask, cache)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
    use_fused_attention: bool = True,
    tie_target_embeddings: bool = True,
) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks: list[EncoderBlock] = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout, use_fused_attention)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks: list[DecoderBlock] = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout, use_fused_attention)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout, use_fused_attention)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, encoder_blocks)
    decoder = Decoder(d_model, decoder_blocks)

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    if tie_target_embeddings:
        projection_layer.proj.weight = tgt_embed.embedding.weight

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
