import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self,width,dropout):
        super().__init__()
        self.path=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*2),nn.GELU(),
                                nn.Dropout(dropout),nn.Linear(width*2,width))
    def forward(self,x): return x+0.5*self.path(x)


class LabelNet(nn.Module):
    def __init__(self,features,labels,width=256,dropout=.1,neighbors=None):
        super().__init__()
        self.backbone=nn.Sequential(nn.Linear(features,width),nn.GELU(),Residual(width,dropout),
                                    Residual(width,dropout),nn.LayerNorm(width))
        self.project=nn.Linear(width,64)
        self.embedding=nn.Parameter(torch.randn(labels,64)*.05)
        self.bias=nn.Parameter(torch.zeros(labels))
        self.graph_gate=nn.Parameter(torch.tensor(-2.0))
        self.register_buffer('neighbors',torch.arange(labels)[:,None] if neighbors is None else neighbors)
        self.use_graph=False
    def forward(self,x):
        e=self.embedding
        if self.use_graph:
            # Sparse top-k neighbors avoid a dense label-by-label multiply per batch.
            e=e+self.graph_gate.sigmoid()*e[self.neighbors].mean(1)
        return self.project(self.backbone(x))@e.T+self.bias


class Linear(nn.Module):
    def __init__(self,features,labels):
        super().__init__(); self.head=nn.Linear(features,labels)
    def forward(self,x): return self.head(x)


def asymmetric_loss(logits,y,gamma_negative=4,clip=.05):
    p=logits.sigmoid()
    negative=(1-p+clip).clamp(max=1)
    loss=y*torch.log(p.clamp_min(1e-8))+(1-y)*negative.log()*((1-negative)**gamma_negative)
    return -loss.mean()


def label_neighbors(y,k=8):
    counts=y.T@y
    scale=y.sum(0).clamp_min(1).sqrt()
    similarity=counts/(scale[:,None]*scale[None,:])
    similarity.fill_diagonal_(-1)
    return similarity.topk(min(k,max(1,y.shape[1]-1)),dim=1).indices


@torch.no_grad()
def predict(model,x,batch=1024):
    model.eval()
    return torch.cat([model(torch.as_tensor(x[i:i+batch])).sigmoid() for i in range(0,len(x),batch)]).numpy()
