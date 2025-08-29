from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from env_config import DEVICE

def _l2norm(x, eps=1e-6): 
    return x / (x.norm(dim=-1, keepdim=True) + eps)

class _Proj(nn.Module):
    def __init__(self, d_in, d_out, p_drop=0.1, use_ln=True):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.ln  = nn.LayerNorm(d_out) if use_ln else nn.Identity()
        self.dp  = nn.Dropout(p_drop)
        nn.init.xavier_uniform_(self.lin.weight); nn.init.zeros_(self.lin.bias)
    def forward(self, x): 
        return self.dp(self.ln(self.lin(x)))

class ConceptRoutingHead(nn.Module):

    def __init__(self, feature_dims: Dict[str, int], num_classes: int,
                 d_concept: int = 256, temperature: float = 0.07,
                 dropout: float = 0.1, use_ln: bool = True, bias_logits: bool = True):
        super().__init__()
        self.names: List[str] = list(feature_dims.keys()) 
        self.M = len(self.names); self.C = num_classes
        self.d_c = d_concept; self.tau = temperature

        self.projs = nn.ModuleDict({ n: _Proj(feature_dims[n], d_concept, dropout, use_ln) for n in self.names })
        # label concepts + readouts
        self.U = nn.Parameter(torch.empty(self.C, d_concept))
        self.O = nn.Parameter(torch.empty(self.C, d_concept))
        nn.init.xavier_uniform_(self.U); nn.init.xavier_uniform_(self.O)
        self.bias = nn.Parameter(torch.zeros(self.C)) if bias_logits else None

    @torch.no_grad()
    def set_temperature(self, tau: float): self.tau = float(tau)

    def forward(self, feats: Dict[str, torch.Tensor]):
        B = next(iter(feats.values())).shape[0]
        Z = torch.stack([ self.projs[n](feats[n]) for n in self.names ], dim=1)  
        Zn, Un = _l2norm(Z), _l2norm(self.U)                                     
        sim = torch.einsum("bmd,cd->bmc", Zn, Un) / max(self.tau, 1e-6)          
        r = F.softmax(sim, dim=1)                                                
        # concept mixtures then readout
        Cmix = torch.einsum("bmc,bmd->bcd", r, Z)                                
        logits = torch.einsum("cd,bcd->bc", self.O, Cmix)                        
        if self.bias is not None: logits = logits + self.bias
        # optional interpretability goodies
        OjZi = torch.einsum("cd,bmd->bmc", self.O, Z)                            
        contrib = OjZi * r
        return logits, {"r": r, "sim": sim, "contrib": contrib, "feat_names": self.names}
