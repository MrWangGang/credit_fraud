import torch
import torch.nn as nn
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, input_dim, output_dim,feature_embedding_dim=64,dropout_rate=0.2,num_attention_heads=8):
        super().__init__()
        self.feature_embedder_linear = nn.Linear(input_dim, input_dim * feature_embedding_dim)
        self.feature_embedder_relu = nn.ReLU()
        self.feature_embedder_dropout = nn.Dropout(dropout_rate)
        self.input_dim = input_dim
        self.feature_embedding_dim = feature_embedding_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_embedding_dim,
            num_heads=num_attention_heads,
            batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.attn_norm = nn.LayerNorm(feature_embedding_dim)
        self.global_average_pooling = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(feature_embedding_dim, output_dim)
    def forward(self, x):
        # 1. 特征嵌入
        x = self.feature_embedder_linear(x)
        x = self.feature_embedder_relu(x)
        x = self.feature_embedder_dropout(x)
        x = x.view(-1, self.input_dim, self.feature_embedding_dim)
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.attn_dropout(attn_output)
        x = self.attn_norm(attn_output)
        x = x.transpose(1, 2)
        x = self.global_average_pooling(x).squeeze(-1)
        return self.output_layer(x)