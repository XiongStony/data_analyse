from torch import  nn
from myMLlib import  RegClsAttention, MultiHeadSelfAttention, RC_CrossAttention, WqAttention
import torch


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
    
class RegClassifier(nn.Module):
    def __init__(self, vec_dim=64, num_heads=4, num_classes=2, attn_dropout=0.001, cls_dropout = 0.001,reg_dropout=0.001):
        super().__init__()
        self.cattn = RegClsAttention(vec_dim, num_heads, attn_dropout=attn_dropout, proj_dropout=0)
        self.rcx = nn.Parameter(torch.zeros(1, 2, vec_dim))
        nn.init.trunc_normal_(self.rcx, std=0.02)
        self.ln = nn.LayerNorm(vec_dim)
        self.classifier = nn.Sequential(
            nn.Linear(vec_dim,14),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(14,num_classes)
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
            nn.Linear(vec_dim,14),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(14,num_classes)
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
            nn.Linear(vec_dim,14),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(14,num_classes)
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

class W2qLastToken(nn.Module):
    def __init__(self, vec_dim=64, num_heads=4, num_classes=2, attn_dropout=0.001, cls_dropout = 0.001,reg_dropout=0.001):
        super().__init__()
        self.attn = WqAttention(vec_dim, num_heads, emb_length=2, attn_dropout=attn_dropout,proj_dropout=0)
        self.pre_ln = nn.LayerNorm(vec_dim)
        self.post_ln = nn.LayerNorm(vec_dim)
        self.classifier = nn.Sequential(
            nn.Linear(vec_dim,14),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(14,num_classes)
        )
        self.regression = RegModule(in_dim=vec_dim + num_classes, arms=24, necks=36, dropout=reg_dropout)

    def forward(self,x):  # token_ids: (B, T)
        x = self.pre_ln(x)
        res = self.attn(x)                   # (B, 2, C)  q/k/v are produced & used here
        x = res + x[:,-1:,:]
        x = self.post_ln(x)
        logits = self.classifier(x[:,0,:])               # (B, num_classes)
        added = torch.cat((logits.detach(),x[:,1,:]),dim=1)
        depth = self.regression(added).squeeze(-1)
        return logits, depth
    
class LastToken(nn.Module):
    def __init__(self, vec_dim=64, num_heads=4, num_classes=1, attn_dropout=0.001, cls_dropout = 0.001,reg_dropout=0.001):
        super().__init__()
        self.attn = WqAttention(vec_dim, num_heads, emb_length=1, attn_dropout=attn_dropout,proj_dropout=0)
        self.ln = nn.LayerNorm(vec_dim)
        self.classifier = nn.Sequential(
            nn.Linear(vec_dim,14),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(14,num_classes)
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
    