"""Split OpenAI CLIP ViT-L/14 (quickgelu) at block K=20: frozen head (embeds + blocks 0..19), trainable tail (blocks 20..23,
ln_post, proj). tail() returns the same 17 x 768 tokens as extract_clip_tokens.py (CLS + 4x4 grid, projected)."""
import torch, torch.nn as nn, torch.nn.functional as F
K = 20

def load_clip(dev, precision="fp32"):
    import open_clip
    m, _, _ = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai", device=dev, precision=precision)
    v = m.visual.eval()
    assert type(v.transformer.resblocks[0].mlp.gelu).__name__ == "QuickGELU"
    return v

def head(v, img):                          # img (N,3,224,224) -> tokens entering block K, (N,257,1024)
    x = v._embeds(img)
    bf = v.transformer.batch_first
    if not bf: x = x.transpose(0, 1)
    for r in v.transformer.resblocks[:K]: x = r(x)
    return x if bf else x.transpose(0, 1)

class Tail(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.blocks = nn.ModuleList(list(v.transformer.resblocks[K:]))
        self.ln_post, self.proj, self.bf = v.ln_post, v.proj, v.transformer.batch_first
        self.ckpt = False

    def forward(self, x):                   # (N,257,1024) -> (N,17,768)
        if not self.bf: x = x.transpose(0, 1)
        for r in self.blocks:
            x = torch.utils.checkpoint.checkpoint(r, x, use_reentrant=False) if (self.ckpt and self.training) else r(x)
        if not self.bf: x = x.transpose(0, 1)
        x = self.ln_post(x)
        cls = x[:, 0] @ self.proj
        g = x[:, 1:].reshape(-1, 16, 16, x.shape[-1]).permute(0, 3, 1, 2)
        g = F.adaptive_avg_pool2d(g.float(), 4).flatten(2).transpose(1, 2).to(x.dtype) @ self.proj
        return torch.cat([cls.unsqueeze(1), g], 1)
