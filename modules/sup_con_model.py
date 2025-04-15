import torch.nn as nn

class SupConModel(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int,
                 embed_dim: int = 768, ):
        super().__init__()
        self.backbone = backbone
        self._cls_head = nn.Linear(in_features=embed_dim, out_features=num_classes)
        self._sup_con_head = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim // 2),
            nn.ReLU(),
            nn.Linear(in_features=embed_dim // 2, out_features=768),
        )


    def forward(self, x, phase='train'):
        x = self.backbone(x)

