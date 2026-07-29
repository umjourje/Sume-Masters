"""model_hybrid.py — W-LSTMix (backbone ORIGINAL) + cabeça de classificação.

O backbone é a classe Model do repositório W-LSTMix (EdgeIntelligenceLab),
importada SEM modificação. A pasta 'models/' desse repositório precisa
estar no sys.path; para não depender do diretório de trabalho, o caminho
é lido da variável de ambiente WLSTMIX_DIR (definida no .env), com
fallbacks para layouts comuns.

    WLSTMIX_DIR=/caminho/ate/o/repo/W-LSTMix     # contém models/W_LSTMix.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# --- localização robusta do pacote 'models' do repo W-LSTMix ---------------
_CANDIDATES = []
_env = os.environ.get("WLSTMIX_DIR")
if _env:
    _CANDIDATES += [_env, os.path.join(_env, "W-LSTMix")]
# fallbacks: mesmo diretório deste arquivo, e um nível acima
_here = Path(__file__).resolve().parent
_CANDIDATES += [str(_here), str(_here.parent),
                str(_here / "W-LSTMix"), str(_here.parent / "W-LSTMix")]

for _c in _CANDIDATES:
    if _c and (Path(_c) / "models" / "W_LSTMix.py").exists():
        if _c not in sys.path:
            sys.path.insert(0, _c)
        break

try:
    from models import W_LSTMix          # backbone original, inalterado
except ModuleNotFoundError as e:         # mensagem acionável
    raise ModuleNotFoundError(
        "Não encontrei o pacote 'models' do repositório W-LSTMix. "
        "Defina WLSTMIX_DIR no .env apontando para a pasta que contém "
        "models/W_LSTMix.py (ex.: o clone de EdgeIntelligenceLab/W-LSTMix), "
        f"ou copie models/ e my_utils/ para {_here}. "
        f"Caminhos testados: {_CANDIDATES}") from e

from config import CFG


class HybridWLSTMix(nn.Module):
    """Backbone de forecasting + cabeça de classificação por timestep.

    forward(trend_in, season_in) -> (trend_pred, season_pred, cls_logits)
      * trend_pred, season_pred: saídas ORIGINAIS do W-LSTMix (horizonte F);
      * cls_logits: um logit por passo do horizonte (F,), da cabeça de
        classificação que recebe a previsão reconstruída + os componentes.
    """

    def __init__(self, device, freeze_backbone: bool = False,
                 pretrained_path=None):
        super().__init__()
        self.device = device
        # RESTRIÇÃO do backbone: o forward faz
        # x.view(batch, backcast_length // embed_dim, embed_dim), então
        # embed_dim TEM de dividir backcast_length. Falhar aqui, com
        # mensagem clara, é muito melhor que o RuntimeError obscuro
        # "shape '[B, k, e]' is invalid for input of size ..." lá dentro.
        if CFG.backcast_length % CFG.embed_dim != 0:
            raise ValueError(
                f"embed_dim ({CFG.embed_dim}) precisa DIVIDIR backcast_length "
                f"({CFG.backcast_length}); caso contrário o reshape interno "
                f"do W-LSTMix trunca ({CFG.backcast_length}//{CFG.embed_dim}"
                f"={CFG.backcast_length // CFG.embed_dim}, "
                f"{CFG.backcast_length // CFG.embed_dim * CFG.embed_dim}"
                f"!={CFG.backcast_length}). Ajuste embed_dim em config.py "
                f"para um divisor de {CFG.backcast_length} "
                f"(ex.: 8 -> 21 patches).")
        self.backbone = W_LSTMix.Model(
            device=device,
            num_blocks_per_stack=CFG.num_blocks_per_stack,
            forecast_length=CFG.forecast_length,
            backcast_length=CFG.backcast_length,
            patch_size=CFG.patch_size,
            num_patches=getattr(CFG, "num_patches",
                                CFG.backcast_length // CFG.patch_size),
            thetas_dim=CFG.thetas_dim,
            hidden_dim=CFG.hidden_dim,
            embed_dim=CFG.embed_dim,
            num_heads=CFG.num_heads,
            ff_hidden_dim=CFG.ff_hidden_dim,
        )
        if pretrained_path:
            # carrega SÓ os pesos do backbone (checkpoint do W-LSTMix puro)
            state = torch.load(pretrained_path, map_location=device)
            self.backbone.load_state_dict(state, strict=False)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        F = CFG.forecast_length
        # Cabeça: recebe [trend_pred, season_pred] concatenados (2F) e produz
        # F logits (um por passo do horizonte). MLP leve, na linha do modelo.
        self.classifier = nn.Sequential(
            nn.Linear(2 * F, CFG.ff_hidden_dim),
            nn.GELU(),
            nn.Linear(CFG.ff_hidden_dim, F),
        )

    def forward(self, trend_in, season_in):
        trend_pred, season_pred = self.backbone(trend_in, season_in)
        feats = torch.cat([trend_pred, season_pred], dim=-1)
        cls_logits = self.classifier(feats)
        return trend_pred, season_pred, cls_logits