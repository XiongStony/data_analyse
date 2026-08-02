from torch import  nn
from myMLlib import  RegClsAttention, MultiHeadSelfAttention, RC_CrossAttention, WqAttention, MutiTaskCAttention,SinusoidalPositionalEncoding
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.nn.functional as F

class ShareAttention(nn.Module):
    def __init__(self, vec_dim, num_heads, emb_length=2, share_feature=16,
                 attn_dropout=0.0, proj_dropout=0.0,
                 backend="math", causal=False, bias=True):
        super().__init__()
        assert vec_dim % num_heads == 0
        self.backend = {
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "math":      SDPBackend.MATH,
            "flash":     SDPBackend.FLASH_ATTENTION,
            "cudnn":     SDPBackend.CUDNN_ATTENTION,
            "overrideable": SDPBackend.OVERRIDEABLE,
        }[backend]
        self.num_heads  = num_heads
        self.head_dim   = vec_dim // num_heads
        self.emb_length = emb_length
        self.share_dim  = share_feature
        self.unique_dim = vec_dim - share_feature
        self.causal     = causal
        self.attn_dropout = float(attn_dropout)

        self.pos_enc  = SinusoidalPositionalEncoding(vec_dim)
        # K/V 对整个序列：一个融合线性层出 K 和 V
        self.Wkv      = nn.Linear(vec_dim, 2 * vec_dim, bias=bias)
        # Q 只对最后一个 token
        self.Wq       = nn.Linear(vec_dim, share_feature + emb_length * self.unique_dim, bias=bias)
        self.out_proj = nn.Linear(vec_dim, vec_dim, bias=bias)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, T, C = x.shape
        H, D, L = self.num_heads, self.head_dim, self.emb_length

        x = self.pos_enc(x)

        # --- Q: 只取最后一个 token，直接 reshape 到多头 ---
        q = self.Wq(x[:, -1, :])  #.view(B, L, H, D).transpose(1, 2)   # (B,H,L,D)
        q = torch.stack((q[:,:C],q[:,self.unique_dim:]),dim=1).view(B, L, H, D).transpose(1, 2)
        # --- K/V: 融合投影 + 一次 reshape + chunk ---
        kv = self.Wkv(x).view(B, T, 2, H, D)                        # (B,T,2,H,D)
        kv = kv.permute(2, 0, 3, 1, 4)                               # (2,B,H,T,D)
        k, v = kv.unbind(0)                                           # (B,H,T,D) each

        with sdpa_kernel(self.backend):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=self.causal
            )                                                         # (B,H,L,D)

        y = y.transpose(1, 2).reshape(B, L, C)                       # (B,L,C)
        y = self.proj_dropout(self.out_proj(y))
        return y
    

class RegModule(nn.Module):
    def __init__(self, in_dim:int, arms:int, necks:int,dropout=1e-3):
        super().__init__()
        self.actfun = nn.GELU()
        self.reg_dropout = nn.Dropout(dropout)
        self.regression1 = nn.Linear(in_dim,arms)
        self.regression2 = nn.Linear(arms,necks)
        self.regression3 = nn.Linear(necks,arms)
        self.regression4 = nn.Linear(arms,1)

    def forward(self,x):
        x    = self.regression1(x)
        x    = self.actfun(x)
        xres = self.regression2(x)
        xres = self.actfun(xres)            # residual layer
        x    = x + self.regression3(xres)   # add residual
        x    = self.actfun(x)
        x    = self.reg_dropout(x)
        x    = self.regression4(x)
        return x
    
class RegModule2(nn.Module):
    def __init__(self, in_dim:int, arms:int, necks:int,dropout=1e-3):
        super().__init__()
        self.actfun = nn.GELU()
        self.reg_dropout = nn.Dropout(dropout)
        self.regression1 = nn.Linear(in_dim,arms)
        self.regression2 = nn.Linear(arms,necks)
        self.regression3 = nn.Linear(necks,1)

    def forward(self,x):
        x    = self.regression1(x)
        x    = self.actfun(x)
        x = self.regression2(x)
        x = self.actfun(x)            # residual layer
        x    = self.reg_dropout(x)
        x    = self.regression3(x)
        return x
    
class RegClassifier(nn.Module):
    def __init__(self, vec_dim=64, num_heads=4, num_classes=2, attn_dropout=0.001, cls_dropout = 0.001,reg_dropout=0.001):
        super().__init__()
        self.cattn = RegClsAttention(vec_dim, num_heads, attn_dropout=attn_dropout, proj_dropout=0)
        self.rcx = nn.Parameter(torch.zeros(1, 2, vec_dim))
        nn.init.trunc_normal_(self.rcx, std=0.02)
        self.ln = nn.LayerNorm(vec_dim)
        self.classifier = nn.Sequential(
            nn.Linear(vec_dim,16),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(16,num_classes)
        )
        self.regression = RegModule(in_dim=vec_dim + num_classes, arms=24, necks=36, dropout=reg_dropout)

    def forward(self,x):  # token_ids: (B, T)
        B, T, C = x.shape
        x = self.ln(x)
        reg_cls = self.rcx.expand(B,2,C)
        x = torch.cat((reg_cls, x), dim=1)
        res = self.cattn(x)                   # (B, C)  q/k/v are produced & used here
        x = x[:,-1:,:] + res

        logits = self.classifier(x[:,0,:])               # (B, num_classes)
        added = torch.cat((logits.detach(),x[:,1,:]),dim=1)
        depth = self.regression(added).squeeze(-1)
        return logits, depth
    
class Traditional(nn.Module):
    def __init__(self, vec_dim=64, num_heads=4, num_classes=2, attn_dropout=0.001, cls_dropout = 0.001,reg_dropout=0.001):
        super().__init__()
        self.attn = MultiHeadSelfAttention(vec_dim, num_heads, attn_dropout=attn_dropout, proj_dropout=0)
        self.ln = nn.LayerNorm(vec_dim)
        self.classifier = nn.Sequential(
            nn.Linear(vec_dim,16),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(16,num_classes)
        )
        self.regression = RegModule(in_dim=vec_dim + num_classes, arms=24, necks=36, dropout=reg_dropout)

    def forward(self,x):  # token_ids: (B, T)
        x = self.ln(x)
        x = self.attn(x).mean(axis=1)                   # (B, C)  q/k/v are produced & used here
        

        logits = self.classifier(x)               # (B, num_classes)
        added = torch.cat((logits.detach(),x),dim=1)
        depth = self.regression(added).squeeze(-1)
        return logits, depth
    
class CrossAtten(nn.Module):
    def __init__(self, vec_dim=64, num_heads=4, num_classes=2, attn_dropout=0.001, cls_dropout = 0.001,reg_dropout=0.001):
        super().__init__()
        self.cattn = RC_CrossAttention(vec_dim, num_heads, attn_dropout=attn_dropout, proj_dropout=0)
        self.rcx = nn.Parameter(torch.zeros(1, 2, vec_dim))
        nn.init.trunc_normal_(self.rcx, std=0.02)
        self.ln = nn.LayerNorm(vec_dim)
        self.classifier = nn.Sequential(
            nn.Linear(vec_dim,16),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(16,num_classes)
        )
        self.regression = RegModule(in_dim=vec_dim + num_classes, arms=24, necks=36, dropout=reg_dropout)

    def forward(self,x):  # token_ids: (B, T)
        B, T, C = x.shape
        reg_cls = self.rcx.expand(B,2,C)
        x = torch.cat((reg_cls, x), dim=1)
        x = self.ln(x)
        x = self.cattn(x)                   # (B, C)  q/k/v are produced & used here
        

        logits = self.classifier(x[:,0,:])               # (B, num_classes)
        added = torch.cat((logits.detach(),x[:,1,:]),dim=1)
        depth = self.regression(added).squeeze(-1)
        return logits, depth
    
# self.regression = RegModule(in_dim=vec_dim + num_classes, arms=24, necks=36, dropout=reg_dropout)
class W2qLastToken(nn.Module):
    def __init__(self, vec_dim=64, num_heads=4, num_classes=2, attn_dropout=0.001, cls_dropout = 0.001,reg_dropout=0.001):
        super().__init__()
        self.attn = WqAttention(vec_dim, num_heads, emb_length=2, attn_dropout=attn_dropout,proj_dropout=0.1,backend="math")
        self.pre_ln = nn.LayerNorm(vec_dim)
        self.post_ln = nn.LayerNorm(vec_dim)
        self.classifier = nn.Sequential(
            nn.Linear(vec_dim,16),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(16,num_classes)
        )
        self.regression = RegModule(in_dim=vec_dim + num_classes, arms=24, necks=36, dropout=reg_dropout)

    def forward(self,x):  # token_ids: (B, T)
        x = self.pre_ln(x)
        res = self.attn(x)                   # (B, 2, C)  q/k/v are produced & used here
        x = res  + x[:,-1:,:]
        x = self.post_ln(x)
        logits = self.classifier(x[:,0,:])               # (B, num_classes)
        added = torch.cat((logits.detach(),x[:,1,:]),dim=1)
        depth = self.regression(added).squeeze(-1)
        return logits, depth

class LastToken(nn.Module):
    def __init__(self, vec_dim=64, num_heads=4, num_classes=1, attn_dropout=0.001, cls_dropout = 0.001,reg_dropout=0.001):
        super().__init__()
        self.attn = WqAttention(vec_dim, num_heads, emb_length=1, attn_dropout=attn_dropout,proj_dropout=0.1)
        self.ln = nn.LayerNorm(vec_dim)
        self.classifier = nn.Sequential(
            nn.Linear(vec_dim,16),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(16,num_classes)
        )
        self.regression = RegModule(in_dim=vec_dim + num_classes, arms=24, necks=36, dropout=reg_dropout)

    def forward(self,x):  # token_ids: (B, T)
        x = self.ln(x)
        res = self.attn(x).squeeze()                   # (B, C)  q/k/v are produced & used here
        x = res + x[:,-1,:]

        logits = self.classifier(x)               # (B, num_classes)
        added = torch.cat((logits.detach(),x),dim=1)
        depth = self.regression(added).squeeze(-1)
        return logits, depth
    
class MTCrossModel(nn.Module):
    def __init__(self, vec_dim=64, num_heads=4, num_classes=2, attn_dropout=0.001, cls_dropout = 0.001,reg_dropout=0.001):
        super().__init__()
        self.attn = MutiTaskCAttention(vec_dim, num_heads, attn_dropout=attn_dropout,proj_dropout=0.1)
        
        D = vec_dim // num_heads
        self.querys = nn.Parameter(torch.zeros(1, num_heads, 2, D))
        nn.init.trunc_normal_(self.querys, std=0.02)

        self.pre_ln = nn.LayerNorm(vec_dim)
        self.post_ln = nn.LayerNorm(vec_dim)
        self.classifier = nn.Sequential(
            nn.Linear(vec_dim,16),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(16,num_classes)
        )
        self.regression = RegModule(in_dim=vec_dim + num_classes, arms=24, necks=36, dropout=reg_dropout)

    def forward(self,x):  # token_ids: (B, T)
        x = self.pre_ln(x)
        x = self.attn(self.querys, x)                   # (B, 2, C)  q/k/v are produced & used here
        # x = res  + x[:,-1:,:]
        # x = self.post_ln(x)
        logits = self.classifier(x[:,0,:])               # (B, num_classes)
        added = torch.cat((logits.detach(),x[:,1,:]),dim=1)
        depth = self.regression(added).squeeze(-1)
        return logits, depth
    
    
class WithoutAttention(nn.Module):
    def __init__(self, vec_dim=64, attn_dropout=0, num_heads=0, num_classes=2, cls_dropout = 0.001,reg_dropout=0.001):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(vec_dim,16),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(16,num_classes)
        )
        self.regression = RegModule(in_dim=vec_dim + num_classes, arms=24, necks=36, dropout=reg_dropout)

    def forward(self,x):  # token_ids: (B, T)
        logits = self.classifier(x)               # (B, num_classes)
        added = torch.cat((logits,x),dim=1)
        depth = self.regression(added).squeeze(-1)
        return logits, depth

class ShareLastToken(nn.Module):
    def __init__(self, vec_dim=64, num_heads=4, num_classes=2, share_feature=16,attn_dropout=0.001, cls_dropout = 0.001,reg_dropout=0.001):
        super().__init__()
        self.attn = ShareAttention(vec_dim, num_heads, emb_length=2, share_feature=share_feature, attn_dropout=attn_dropout,proj_dropout=0.1,backend="math")
        self.pre_ln = nn.LayerNorm(vec_dim)
        self.post_ln = nn.LayerNorm(vec_dim)
        self.classifier = nn.Sequential(
            nn.Linear(vec_dim,16),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(16,num_classes)
        )
        self.regression = RegModule(in_dim=vec_dim + num_classes, arms=24, necks=36, dropout=reg_dropout)

    def forward(self,x):  # token_ids: (B, T)
        res = self.attn(self.pre_ln(x))                   # (B, 2, C)  q/k/v are produced & used here
        x = res  + x[:,-1:,:]
        x = self.post_ln(x)
        logits = self.classifier(x[:,0,:])               # (B, num_classes)
        added = torch.cat((logits.detach(),x[:,1,:]),dim=1)
        depth = self.regression(added).squeeze(-1)
        return logits, depth