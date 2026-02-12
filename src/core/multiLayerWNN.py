from pyexpat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import List, Tuple, Optional, List

from src.core.wnnLutLayer import WNNLUTLayer
from src.tools.utils import get_exit1_features


class MultiLayerWNN(nn.Module):
    def __init__(
        self,
        in_bits: int,
        num_classes: int,
        lut_input_size: int = 6,
        hidden_luts=(2000, 1000),
        mapping=None,
        tau: float = 1.0,
        exit_tau: float = 1.0,
    ):
        super().__init__()
        self.tau = tau
        self.exit1_use_norm = False  # 是否對 exit1 features 做 mu/sigma normalization；建議預設 False，除非你確定要用且已經準備好 mu/sigma buffer

        layers = []
        prev_bits = in_bits

        self.layer_in_bits = []   # input bits per layer
        self.layer_out_luts = []  # number of LUTs per layer

        # for early exit start
        #self.exit1_classifier = nn.Linear(hidden_luts[0], num_classes, bias=False)
        self.exit_tau = exit_tau  # 或者單獨設


        # in MultiLayerWNN.__init__()
        self.exit1_classifier = None   # nn.Linear(K, C)
        '''self.register_buffer("exit1_keep_idx", torch.empty(0, dtype=torch.long))
        self.register_buffer("exit1_mu", torch.empty(0))
        self.register_buffer("exit1_sigma", torch.empty(0))'''



        for i, n_lut in enumerate(hidden_luts):
            # Use mapping only for the first layer
            layer_mapping = mapping if i == 0 else None

            layers.append(
                WNNLUTLayer(
                    in_bits=prev_bits,
                    num_luts=n_lut,
                    lut_input_size=lut_input_size,
                    mapping=layer_mapping # <--- PASS IT HERE
                )
            )
            self.layer_in_bits.append(prev_bits)
            self.layer_out_luts.append(n_lut)
            prev_bits = n_lut

        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(prev_bits, num_classes, bias=False)

        # for hidden pruning
        self.register_buffer("keep_idx", torch.empty(0, dtype=torch.long))

    # helper
    def enable_exit1(self, K: int, num_classes: int, bias: bool = True, exit_tau: float = 1.0, device=None):
        self.exit_tau = float(exit_tau)
        self.exit1_classifier = nn.Linear(K, num_classes, bias=bias)
        if device is not None:
            self.exit1_classifier = self.exit1_classifier.to(device)

    def disable_exit1(self):
        self.exit1_classifier = None
        # keep_idx / mu / sigma 你可以選擇保留或清掉

    def forward(self, x_bits, return_hidden: bool = False):
        h = x_bits
        for layer in self.layers:
            h = layer(h)
        
        if self.keep_idx.numel() > 0:
            h_used = h[:, self.keep_idx]
        else:
            h_used = h

        logits = self.classifier(h_used) / self.tau

        if return_hidden:
            return logits, h
        else:
            return logits

    def forward_with_all_hidden(self, x_bits: torch.Tensor):
        """
        Return:
          logits: [B, C]
          h_list: list of length L, where the l-th element is [B, num_luts_l]
        """
        h_list = []
        h = x_bits
        for layer in self.layers:
            h = layer(h)
            h_list.append(h)
        if self.keep_idx.numel() > 0:
            h_used = h[:, self.keep_idx]
        else:
            h_used = h
        logits = self.classifier(h_used) / self.tau
        return logits, h_list
    

    def forward_with_all_hidden_and_exits(self, x_bits: torch.Tensor):
        h_list = []
        h = x_bits
        exit1_logits = None

        for li, layer in enumerate(self.layers):
            h = layer(h)
            h_list.append(h)

            if li == 0 and (self.exit1_classifier is not None):
                h_exit = h
                if self.exit1_keep_idx.numel() > 0:
                    h_exit = h_exit[:, self.exit1_keep_idx]

                # optional normalization (recommended if buffers exist)
                if self.exit1_use_norm:
                    if (self.exit1_mu.numel() > 0) and (self.exit1_sigma.numel() > 0):
                        h_exit = (h_exit - self.exit1_mu) / self.exit1_sigma

                exit1_logits = self.exit1_classifier(h_exit) / self.exit_tau

        # final logits
        if self.keep_idx.numel() > 0:
            h_used = h[:, self.keep_idx]
        else:
            h_used = h
        final_logits = self.classifier(h_used) / self.tau

        return final_logits, exit1_logits, h_list



    def forward_with_all_hidden_and_exits_g1(self, x_bits: torch.Tensor):
        h_list = []

        # layer 1
        h1 = self.layers[0](x_bits)
        h_list.append(h1)

        exit1_logits = None
        if self.exit1_classifier is not None:
            h1_exit = h1
            if (self.exit1_keep_idx is not None) and (self.exit1_keep_idx.numel() > 0):
                h1_exit = h1_exit[:, self.exit1_keep_idx]

            if self.exit1_use_norm:
                if (self.exit1_mu is not None) and (self.exit1_sigma is not None) and (self.exit1_mu.numel() > 0) and (self.exit1_sigma.numel() > 0):
                    h1_exit = (h1_exit - self.exit1_mu) / (self.exit1_sigma + 1e-6)

            exit1_logits = self.exit1_classifier(h1_exit) / self.exit_tau

        # final branch (stop grad to layer1)
        h2 = self.layers[1](h1.detach())
        h_list.append(h2)

        if self.keep_idx.numel() > 0:
            h2_used = h2[:, self.keep_idx]
        else:
            h2_used = h2

        final_logits = self.classifier(h2_used) / self.tau
        return final_logits, exit1_logits, h_list
    


    # -------------------------
    # Forward helpers
    # -------------------------
    @torch.no_grad()
    def _exit_gate_margin(self, exit_logits: torch.Tensor, thr: float):
        """
        Return:
          exit_mask: [B] bool
          margin: [B] float (top1 - top2 on logits)
        """
        top2 = torch.topk(exit_logits, k=2, dim=-1).values  # [B, 2]
        margin = top2[:, 0] - top2[:, 1]
        exit_mask = margin > thr
        return exit_mask, margin
    
    def forward_g2(self, x_bits: torch.Tensor):
        """
        G2: standard forward (no detach), because layer1 will be frozen anyway.
        Return: final_logits, exit1_logits
        """
        h1 = self.layers[0](x_bits)
        h1_exit = get_exit1_features(self, h1)
        exit1_logits = self.exit1_classifier(h1_exit) / self.exit_tau

        h2 = self.layers[1](h1)
        final_logits = self.classifier(h2) / self.tau
        return final_logits, exit1_logits
    

    def forward_g2_with_mask(self, x_bits: torch.Tensor, thr: float):
        # layer1
        h1 = self.layers[0](x_bits)

        # exit logits
        h1_exit = h1
        if self.exit1_keep_idx is not None and self.exit1_keep_idx.numel() > 0:
            h1_exit = h1_exit[:, self.exit1_keep_idx]

        if self.exit1_use_norm:
            if self.exit1_mu is not None and self.exit1_mu.numel() > 0:
                h1_exit = (h1_exit - self.exit1_mu) / self.exit1_sigma

        exit1_logits = self.exit1_classifier(h1_exit) / self.exit_tau

        # gate / mask
        top2 = torch.topk(exit1_logits, k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        exit_mask = margin > thr

        # final branch ALWAYS computed (this is key)
        h2 = self.layers[1](h1)              # 不要 detach；G2 freeze layer1 就好
        final_logits = self.classifier(h2) / self.tau

        return final_logits, exit1_logits, exit_mask



    def forward_g3(self, x_bits: torch.Tensor):
        """
        G3: joint training forward, no detach.
        Return: final_logits, exit1_logits
        """
        h1 = self.layers[0](x_bits)
        h1_exit = get_exit1_features(self, h1)
        exit1_logits = self.exit1_classifier(h1_exit) / self.exit_tau

        h2 = self.layers[1](h1)
        final_logits = self.classifier(h2) / self.tau
        return final_logits, exit1_logits





import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional, Dict, Any, Tuple

def _prealloc_exit_buffers_from_ckpt(model, sd, device):
    # exit1_keep_idx
    if "exit1_keep_idx" in sd:
        t = sd["exit1_keep_idx"].to(device)
        # 如果已經存在同名 buffer，直接覆蓋成同 shape tensor
        # 注意：這裡要用 register_buffer 需要先刪掉舊的
        if "exit1_keep_idx" in model._buffers:
            del model._buffers["exit1_keep_idx"]
        model.register_buffer("exit1_keep_idx", torch.empty_like(t))

    if "exit1_mu" in sd:
        t = sd["exit1_mu"].to(device)
        if "exit1_mu" in model._buffers:
            del model._buffers["exit1_mu"]
        model.register_buffer("exit1_mu", torch.empty_like(t))

    if "exit1_sigma" in sd:
        t = sd["exit1_sigma"].to(device)
        if "exit1_sigma" in model._buffers:
            del model._buffers["exit1_sigma"]
        model.register_buffer("exit1_sigma", torch.empty_like(t))


def build_model_from_configs(backbone_config: Dict[str, Any],
                             exit_config: Optional[Dict[str, Any]],
                             device):
    # ---- build backbone ----
    cfg = backbone_config
    model = MultiLayerWNN(
        in_bits=cfg["in_bits"],
        num_classes=cfg["num_classes"],
        lut_input_size=cfg["lut_input_size"],
        hidden_luts=tuple(cfg["hidden_luts"]),
        mapping=cfg.get("mapping", None),
        tau=float(cfg.get("tau", 1.0)),
    ).to(device)
    print("[A] after ctor:", list(dict(model.named_buffers()).keys()))
    
    # Ensure these exist (recommended: put them in __init__ instead)
    if not hasattr(model, "exit_tau"):
        model.exit_tau = 1.0
    if not hasattr(model, "exit1_classifier"):
        model.exit1_classifier = None

    # ---- build exit (optional) ----
    ex = exit_config or {}
    if ex.get('use_norm', True):
        model.exit1_use_norm = True
        
    if ex.get("enabled", False):
        head_type = ex.get("head_type", "linear")
        K = int(ex["K"])
        model.exit_tau = float(ex.get("exit_tau", 1.0))

        # keep_idx 是 long
        model.register_buffer("exit1_keep_idx", torch.zeros(K, dtype=torch.long, device=device))
        model.register_buffer("exit1_mu", torch.zeros(K, device=device))
        model.register_buffer("exit1_sigma", torch.ones(K, device=device))


        if head_type == "linear":
            model.exit1_classifier = nn.Linear(K, cfg["num_classes"], bias=True).to(device)
        else:
            raise ValueError("Unsupported head_type={}".format(head_type))
        

    return model


def save_ckpt(path: str,
              model,
              backbone_config: Dict[str, Any],
              exit_config: Optional[Dict[str, Any]] = None,
              extra: Optional[Dict[str, Any]] = None):
    ckpt = {
        "format_version": 1,
        "backbone_config": deepcopy(backbone_config),
        "exit_config": deepcopy(exit_config) if exit_config is not None else None,
        "model_state": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(ckpt, path)


def load_ckpt(path: str,
              device,
              backbone_config_fallback: Optional[Dict[str, Any]] = None):
    obj = torch.load(path, map_location=device)


    # legacy: pure state_dict
    if isinstance(obj, dict) and ("model_state" not in obj) and ("backbone_config" not in obj):
        if backbone_config_fallback is None:
            raise ValueError("Legacy state_dict ckpt needs backbone_config_fallback.")
        ckpt = {
            "format_version": 0,
            "backbone_config": deepcopy(backbone_config_fallback),
            "exit_config": None,
            "model_state": obj,
            "extra": {"legacy": True},
        }
    else:
        ckpt = obj

    print("[load] exit_config:", ckpt.get("exit_config", None))

    bb_cfg = ckpt.get("backbone_config", None)
    if bb_cfg is None:
        raise ValueError("Checkpoint missing backbone_config.")

    ex_cfg = ckpt.get("exit_config", None)

    model = build_model_from_configs(bb_cfg, ex_cfg, device=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)

    #print(model.exit1_keep_idx)
    print("[load_ckpt] missing:", missing)
    print("[load_ckpt] unexpected:", unexpected)

    return model, bb_cfg, ex_cfg, ckpt.get("extra", {})

