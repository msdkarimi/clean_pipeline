import torch.nn as nn
import torch

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

        if phase == 'train':
            _bs = x.shape[0]//2
            _sup_con_logits = torch.nn.functional.normalize(self._sup_con_head(x))

            f1, f2 =  torch.split(_sup_con_logits, [_bs, _bs], dim=0)
            sup_con_logits = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            _, cls_ftr = torch.split(x, [_bs, _bs], dim=0)
            cls_logit = self._cls_head(cls_ftr.detach())
        else:
            sup_con_logits = torch.nn.functional.normalize(self._sup_con_head(x))
            cls_logit = self._cls_head(x)

        return sup_con_logits, cls_logit








