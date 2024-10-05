import torch
from torch import nn
from einops import rearrange
from .layers.LabelEmbedder import LabelEmbedder
from transformers import AutoModel
from .layers.SinusoidalPositionEmbeddings import SinusoidalPositionEmbeddings


class Denoiser(nn.Module):
    def __init__(self, esm_model_path, denoiser_embedding, denoiser_mlp):
        super(Denoiser, self).__init__()
        self.denoiser_embedding = denoiser_embedding
        self.denoiser_mlp = denoiser_mlp
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(self.denoiser_embedding * 2),
            nn.Linear(self.denoiser_embedding * 2, self.denoiser_embedding * 4),
            nn.SiLU(),
            nn.Linear(self.denoiser_embedding * 4, self.denoiser_embedding * 4),
        )
        # 3个类别，以0.2的概率将类别遮掩掉对应类别
        self.label_emb = LabelEmbedder(3, self.denoiser_embedding * 4, 0.2)

        esm_model = AutoModel.from_pretrained(esm_model_path, trust_remote_code=True, output_hidden_states=True)
        esm_attention_list = []
        for i, layer in enumerate(esm_model.encoder.layer):
            esm_attention_list.append(layer.attention)
        self.esm_attention_list = nn.ModuleList(esm_attention_list)

        time_emb_proj_list = []
        for i, layer in enumerate(self.esm_attention_list):
            time_emb_proj_list.append(nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.denoiser_embedding * 4, self.denoiser_embedding * 2)
            ))
        self.time_emb_proj_list = nn.ModuleList(time_emb_proj_list)

        self.norm_before = nn.LayerNorm(self.denoiser_embedding, eps=1e-6)
        self.norm_after = nn.LayerNorm(self.denoiser_embedding, eps=1e-6)

        mlp_list = [
            nn.Linear(self.denoiser_embedding, self.denoiser_mlp[0]),
            nn.LayerNorm(self.denoiser_mlp[0], eps=1e-6),
            nn.GELU(),
        ]
        for index in range(len(self.denoiser_mlp) - 1):
            mlp_list.append(nn.Linear(self.denoiser_mlp[index], self.denoiser_mlp[index + 1]))
            mlp_list.append(nn.LayerNorm(self.denoiser_mlp[index + 1], eps=1e-6))
            mlp_list.append(nn.GELU())
        mlp_list.append(nn.Linear(self.denoiser_mlp[-1], self.denoiser_embedding))
        self.mlp = nn.Sequential(*mlp_list)

    def forward(self, x, time, y, attention_mask=None, return_attn_matrix=False):
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].expand(attention_mask.shape[0], 1, attention_mask.shape[1], attention_mask.shape[1])

        # 时间步位置嵌入
        time_emb = self.time_emb(time)
        # 条件嵌入
        label_emb = self.label_emb(y, self.training)
        c = torch.add(time_emb, label_emb)

        x = self.norm_before(x)
        attn_output = x.clone()

        attn_matrix_list = []
        for i, (time_emb_layer, attention_layer) in enumerate(zip(self.time_emb_proj_list, self.esm_attention_list)):
            mlp_c = time_emb_layer(c)
            mlp_c = rearrange(mlp_c, "b c -> b 1 c")
            scale, shift = mlp_c.chunk(2, dim=-1)

            # 缩放因子（相乘）和偏差调整（相加）
            attn_output = torch.add(attn_output * (scale + 1.), shift)

            # attn_matrix: [batch_size, num_heads, sequence_length, sequence_length]
            attn_output, attn_matrix = attention_layer(attn_output, attention_mask=attention_mask, output_attentions=True)
            attn_matrix_list.append(attn_matrix)

        x = self.norm_after(torch.add(x, attn_output))

        x = self.mlp(x)

        if return_attn_matrix:
            return x, attn_matrix_list
        else:
            return x
