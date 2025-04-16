import torch
import torch.nn as nn


class Adapter_Layer(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=0.25, norm_layer=nn.LayerNorm, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm = norm_layer(embed_dim)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim, bias=False),
            nn.Sigmoid()
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
        )

        # for m in self.modules():
        #     if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # x -> （B, H, W, C）-> （B, C, H, W）
        x = x.permute(0, 3, 1, 2)
        B, C, _, _ = x.size()
        x_channel = self.channel(self.avg_pool(x).view(B, C)).view(B, C, 1, 1) * x
        x_spatial = self.spatial(x_channel)

        if self.skip_connect:
            x = x + x_spatial
        else:
            x = x_spatial
        # （B, C, H, W） -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        return self.norm(x)


class Adapter (nn.Module):
    def __init__(self, input_dim, residual=False, ln_input_output=False, dropout=0.1):
        super(Adapter, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.ln_input_output = ln_input_output
        self.residual = residual

        if ln_input_output:
            self.layer_norm_input = nn.LayerNorm(input_dim)
            self.layer_norm_output = nn.LayerNorm(input_dim)
            self.scale = nn.Parameter(torch.ones(1))

        self.down_proj = nn.Linear(input_dim, input_dim//2)

        self.non_linear_func = nn.ReLU()

        self.up_proj = nn.Linear(input_dim//2, input_dim)


    def forward(self, x):
        if self.ln_input_output:
            output = self.forward_with_ln(x)
        else:
            output = self.forward_without_ln(x)

        if self.residual:
            output += x
        else:
            output

        return output

    def forward_with_ln(self, x):
        down = self.down_proj(self.layer_norm_input(x))
        down = self.non_linear_func(down)
        drop_out = self.dropout(down)
        up = self.up_proj(drop_out)
        up = up * self.scale
        output = self.layer_norm_output(up)
        return output

    def forward_without_ln(self, x):
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        drop_out = self.dropout(down)
        up = self.up_proj(drop_out)
        return up