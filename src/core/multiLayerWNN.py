import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import List, Tuple, Optional, List
from src.core.wisard import LearnableConnTopK, build_conn0_from_buckets, build_conn0_from_buckets_2, build_conn0_from_buckets_3, build_conn0_from_buckets_4, build_conn0_hybrid_from_buckets, build_conn0_rgb_thermo_sobel_v2
from src.core.wnnLutLayer import WNNLUTLayer
from src.core.wnnLutLayerSoftConn import WNNLUTLayerSoftConn
from src.dataio.encode import bucket_id_from_global_idx, bucket_id_from_global_idx_rgb_th_sobel2
from src.exit.ckpt_exit import normalize_exit_cfg_list
from src.tools.utils import get_exit1_features, make_dropout_schedule, summarize_conn0


class MultiLayerWNN(nn.Module):
    """
    Quasi-Weightless Transformers (QuWeiT) Multi-Layer Neural Network with DWB (Differentiable Weightless Blocks).
    
    Paper: "Shrinking the Giant: Quasi-Weightless Transformers for Low Energy Inference"
    arXiv:2411.01818v1
    
    Architecture Overview:
    =====================
    This model implements energy-efficient LUT-based neural networks that replace transformer MLPs with
    Differentiable Weightless Blocks (DWB). Each layer follows the pattern:
    
        Input bits → LUT table lookup → Binary outputs → Conditional summation → Real-valued features
    
    Layer Configuration:
    -------------------
    Layer 0 (Hard-wired connections):
        - Fixed mapping: RGB image → Thermometer encoding → Sobel features
        - Input: in_bits (2048 for 32×32×8bit CIFAR)
        - LUT table: [num_luts[0], 2^k] where k=lut_input_size[0]=9
        - Output: [B, num_luts[0]] binary values
        - With DWB: [B, encoded_values_out_dim] real-valued features via conditional summation
    
    Layer 1 (Learnable connections):
        - Learnable bit selection via Gumbel-softmax
        - Input: encoded_values_out_dim (from Layer 0) or num_luts[0] (without DWB)
        - LUT table: [num_luts[1], 2^k] where k=lut_input_size[1]=5
        - Output: [B, num_luts[1]] binary values
        - With DWB: [B, encoded_values_out_dim] real-valued features via conditional summation
    
    Classifier:
        - Linear projection from final layer outputs to class logits
        - Input: encoded_values_out_dim (with DWB) or num_luts[1] (without DWB)
        - Output: [B, num_classes]
    
    Differentiable Weightless Block (DWB) Mechanism:
    ================================================
    Instead of standard neural network layers with weight matrices, DWB uses:
    
    1. **Conditional Summation** (key innovation):
        binary_out = (lut_output > 0).float()           # [B, num_luts]
        output = (binary_out.unsqueeze(-1) × encoded_values).sum(dim=1)  # [B, encoded_dim]
    
    2. **Benefits**:
        - No multiplications: addition-only operations (efficient hardware)
        - Real-valued outputs: smooth gradient flow for training
        - Learnable features: encoded_values optimized via backprop
        - Parameter reduction: ~66% elimination vs standard MLPs
        - Energy efficiency: ~2.2x improvement, 55% MAC reduction
    
    3. **Training Dynamics**:
        - Extended Finite Differentiation enables end-to-end gradients
        - Gumbel-softmax for learning connection patterns
        - 3-phase temperature schedule: exploration → transition → commitment
        - Entropy regularization guides learning
    
    Parameters:
    -----------
    in_bits: int
        Input dimension (e.g., 2048 for CIFAR-10 with thermometer encoding)
    num_classes: int
        Number of output classes (e.g., 10 for CIFAR)
    lut_input_size: int
        Default LUT input size (bits per LUT). Overridden by lut_input_size_list if provided.
    lut_input_size_list: Optional[List[int]]
        Per-layer LUT input sizes (e.g., [9, 5] for layers 0 and 1)
    hidden_luts: tuple
        Number of LUTs per layer (e.g., (2000, 1000))
    mapping: Optional
        Pre-computed bit-to-LUT mapping (None = use built-in thermo+sobel for layer 0)
    tau: float
        Temperature scaling for final classifier logits (default: 1.0)
    exit_tau: float
        Temperature scaling for early exit heads (default: 1.0)
    dropout_p: float or str
        Dropout probability schedule (e.g., "0.2,0.15" for per-layer rates)
    encoded_values_out_dim: Optional[int]
        **DWB FEATURE**: Dimension of encoded values per LUT layer.
        - If None: Standard LUT outputs (binary), no DWB
        - If int: Enable DWB with this output dimension
        - Example: encoded_values_out_dim=8 → [B, 8] output per DWB layer
    
    Example Usage:
    ---------------
    # Without DWB (standard WNN)
    model = MultiLayerWNN(in_bits=2048, num_classes=10, hidden_luts=(2000, 1000))
    x = torch.randn(32, 2048)  # Batch of 32 encoded images
    logits = model(x)  # [32, 10]
    
    # With DWB (Quasi-Weightless Transformers)
    model = MultiLayerWNN(
        in_bits=2048, 
        num_classes=10, 
        hidden_luts=(2000, 1000),
        encoded_values_out_dim=8,  # Enable DWB with 8-D features
    )
    x = torch.randn(32, 2048)
    logits = model(x)  # [32, 10], with internal transformations via 8-D representations
    
    References:
    -----------
    - DWB paper: arXiv:2411.01818v1
    - Prior DWN work: Bacellar et al. 2024 (Extended Finite Differentiation)
    - Thermometer encoding: Widely used in weightless networks
    - Gumbel-softmax: Jang et al. 2017 (differentiable sampling)
    """
    def __init__(
        self,
        in_bits: int,
        num_classes: int,
        lut_input_size: int = 6,
        lut_input_size_list: Optional[List[int]] = None,
        hidden_luts=(2000, 1000),
        mapping=None,
        tau: float = 1.0,
        exit_tau: float = 1.0,
        dropout_p=0.0,
        encoded_values_out_dim: Optional[int] = None,
    ):
        super().__init__()
        self.tau = tau
        self.exit1_use_norm = False  # 是否對 exit1 features 做 mu/sigma normalization；建議預設 False，除非你確定要用且已經準備好 mu/sigma buffer
        self.encoded_values_out_dim = encoded_values_out_dim

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
        # 每層 dropout schedule
        # inside MultiLayerWNN.__init__ after defining layers=[], prev_bits=in_bits, etc.

        # 1) dropout schedule
        drop_ps = make_dropout_schedule(dropout_p, num_layers=len(hidden_luts))
        # 如果你真的要手動指定，建議用參數傳進來，而不是這裡 hardcode
        drop_ps = [0.2, 0.15]
        init_std_list = [0.05, 0.01]

        # 2) lut_input_size per-layer
        if lut_input_size_list is not None:
            assert len(lut_input_size_list) == len(hidden_luts), \
                f"lut_input_size_list len {len(lut_input_size_list)} must match hidden_luts len {len(hidden_luts)}"
        else:
            lut_input_size_list = [lut_input_size] * len(hidden_luts)




        # layer0 (existing hard LUT)
        conn0 = build_conn0_rgb_thermo_sobel_v2(
                    num_luts=hidden_luts[0],
                    k=lut_input_size_list[0],
                    frac_thermo=0.22,
                    frac_sobel=0.33,
                    patch=(6, 6),
                    seed=42,
                    device="cpu",
                    sobel_jitter_p=0.0,
                    sobel_global_frac=0.24,
                ).to(torch.long).cpu()
        
        layer0 = WNNLUTLayer(
            in_bits=in_bits, num_luts=hidden_luts[0], lut_input_size=lut_input_size_list[0],
            conn_idx=conn0, binarize_input=True, dropout_p=0.2,
            init_std=0.05,
            encoded_values_out_dim=encoded_values_out_dim  # DWB support
        )

        # Determine layer1 input dimension based on whether DWB is enabled
        layer1_in_bits = encoded_values_out_dim if encoded_values_out_dim is not None else hidden_luts[0]
        
        # layer1 (learnable conn + soft LUT train)
        layer1 = WNNLUTLayerSoftConn(
            in_bits=layer1_in_bits, num_luts=hidden_luts[1], k=lut_input_size_list[1],
            M=128, init_std=0.02, dropout_p=0.0,
            seed=42,
            use_gumbel=True, gumbel_tau=2.0,
            binarize_mode="sign",   # 你現在 layer1 用 sign 效果最好
            encoded_values_out_dim=encoded_values_out_dim  # DWB support
        )

        self.layers = nn.ModuleList([layer0, layer1])
        
        # Determine classifier input dimension based on whether DWB is enabled on final layer
        classifier_in_dim = encoded_values_out_dim if encoded_values_out_dim is not None else hidden_luts[1]
        self.classifier = nn.Linear(classifier_in_dim, num_classes, bias=False)
        self.register_buffer("keep_idx", torch.empty(0, dtype=torch.long))

        '''for i, n_lut in enumerate(hidden_luts):
            k_i = int(lut_input_size_list[i])

            # layer0: input is bitplane (0/1) so binarize_input=True is OK
            # later: your LUT output is real-valued -> binarize_input=False
            #binarize_input = True if i == 0 else False

            if i == 0:
                # NOTE: conn0 must be built using k_i (not the global lut_input_size)
                conn0 = build_conn0_rgb_thermo_sobel_v2(
                    num_luts=n_lut,
                    k=k_i,
                    frac_thermo=0.22,
                    frac_sobel=0.33,
                    patch=(6, 6),
                    seed=42,
                    device="cpu",
                    sobel_jitter_p=0.0,
                    sobel_global_frac=0.24,
                ).to(torch.long).cpu()
                conn_idx = conn0
                learnable_conn = None
                binarize_input = True
            else:
                conn_idx = None  # later layers random wiring
                binarize_input = False

                learnable_conn = LearnableConnTopK(
                    num_luts=n_lut,
                    in_bits=prev_bits,   # layer1 input dim = layer0 num_luts
                    k=k_i,
                    M=64,
                    seed=42,
                    use_gumbel=False,    # 先關掉，穩定後再試
                    gumbel_tau=1.0,
                )

            layer = WNNLUTLayer(
                in_bits=prev_bits,
                num_luts=n_lut,
                lut_input_size=k_i,        # ✅ 這行就是修正 A 的核心
                conn_idx=conn_idx,
                mapping=None,
                binarize_input=binarize_input,
                dropout_p=drop_ps[i],
                init_std=init_std_list[i],
                # 你有跑 adaptive threshold，但目前 forward 沒用到它；
                # 先保持預設不影響現有結果
                learnable_conn=learnable_conn,  # ✅ 新增
            )

            layers.append(layer)

            if i == 0:
                print("layer0 conn idx preview:", layers[0].conn_idx[:5])
                print("layer0 conn idx max:", layers[0].conn_idx.max().item(), "in_bits:", layers[0].in_bits)

            self.layer_in_bits.append(prev_bits)
            self.layer_out_luts.append(n_lut)
            prev_bits = n_lut

            self.layers = nn.ModuleList(layers)
            self.classifier = nn.Linear(prev_bits, num_classes, bias=False)
            # for hidden pruning
            self.register_buffer("keep_idx", torch.empty(0, dtype=torch.long))

            for i, layer in enumerate(self.layers):
                print(f"layer{i}: k={layer.lut_input_size}, in_bits={layer.in_bits}, num_luts={layer.num_luts}")'''

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
        #b = (self.layers[0].conn_idx.view(-1) % 8).cpu()
        #print("bitplane hist:", torch.bincount(b, minlength=8).float() / b.numel())
        for layer in self.layers:
            h = layer(h)
            #print(f"after layer {layer}: h shape {h.shape}, min {h.min().item()}, max {h.max().item()}, mean {h.mean().item()}, std {h.std().item()}")
        
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

'''def load_ckpt_v2(path, device):
    ckpt = torch.load(path, map_location=device)
    backbone_cfg = ckpt["backbone_cfg"]
    payload_exit_cfg = ckpt.get("exit_cfg", [])
    exit_cfg_list = normalize_exit_cfg_list(payload_exit_cfg)  # <-- 轉成 ExitConfig list

    # 你原本的 init 邏輯維持：用 backbone_cfg 建 model
    model = MultiLayerWNN(**backbone_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    extra = ckpt.get("extra", {})
    return model, backbone_cfg, exit_cfg_list, extra'''


import torch
from typing import Any, Dict, Optional, Tuple, List

def load_ckpt_v2(
    path: str,
    device,
    build_model_fn,              # build_model_from_configs or your wrapper
    build_exit_heads_fn,         # build_exit_heads_from_cfg (below)
    load_exits: bool = True,
):
    """
    Returns:
      model, backbone_cfg, exit_cfg_list(payload), exit_heads, extra
    """
    ckpt = torch.load(path, map_location=device)

    # ---- configs ----
    backbone_cfg = ckpt["backbone_cfg"]
    exit_cfg_payload = ckpt.get("exit_cfg", None)  # list of payload dicts
    extra = ckpt.get("extra", {})

    # ---- build backbone-only model ----
    # 這裡最重要：不要把 exit_cfg 丟進 build_model（避免 model 裡 already built exits）
    model = build_model_fn(backbone_cfg, ex_cfg=None, device=device).to(device)

    # ---- load backbone weights ----
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print("[load_ckpt_v2] backbone missing:", missing)
    print("[load_ckpt_v2] backbone unexpected:", unexpected)

    # ---- optionally load exits ----
    exit_heads = None
    if load_exits and exit_cfg_payload is not None:
        # 1) payload -> cfg objects (if you want)
        exit_cfg_list = exit_cfg_payload  # 先用 payload 也行（下面 head builder 支援 dict）
        # 2) build heads
        C = backbone_cfg["num_classes"] if "num_classes" in backbone_cfg else backbone_cfg["dataset_meta"]["num_classes"]
        exit_heads = build_exit_heads_fn(exit_cfg_list, num_classes=C, device=device)
        # 3) load head states
        exits_sd: List[Dict[str, torch.Tensor]] = ckpt["exits_state_dict"]
        assert len(exits_sd) == len(exit_heads), "exits_state_dict length mismatch"
        for h, sd in zip(exit_heads, exits_sd):
            h.load_state_dict(sd, strict=True)

        return model, backbone_cfg, exit_cfg_payload, exit_heads, extra

    return model, backbone_cfg, exit_cfg_payload, None, extra




def save_ckpt_v2(path, model, exit_heads, backbone_cfg, exit_cfg_list=None, extra=None):
    exit_cfg_list = normalize_exit_cfg_list(exit_cfg_list)
    payload_exit_cfg = [ec.to_payload() for ec in exit_cfg_list]
    exit_dict = []
    for exit in exit_heads:
        exit_dict.append(exit.state_dict())

    ckpt = {
        "model_state_dict": model.state_dict(),
        "exits_state_dict": exit_dict,
        "backbone_cfg": backbone_cfg,
        "exit_cfg": payload_exit_cfg,   # <-- 永遠是 payload list
        "extra": extra or {},
    }
    torch.save(ckpt, path)


import os
import shutil
import torch

def save_best_checkpoint_atomic(
    path_out: str,
    model: torch.nn.Module,
    best_val_acc: float,
    epoch: int,
    optimizer=None,
    scheduler=None,
    extra: dict = None,
):
    """
    Save checkpoint to a temp file then atomically replace `path_out`.
    This avoids corrupting `path_out` if interrupted during write.
    """
    tmp_path = path_out + ".tmp"

    payload = {
        "epoch": epoch,
        "best_val_acc": float(best_val_acc),
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if extra is not None:
        payload["extra"] = extra

    # write temp
    torch.save(payload, tmp_path)

    # atomic replace (best effort cross-platform)
    os.replace(tmp_path, path_out)


# src/ckpt/loaders.py
from typing import Any, Dict, Tuple, Optional
import torch

def build_backbone_only(backbone_cfg: dict, device):
    cfg = dict(backbone_cfg)

    # 常見非建模欄位先移除（保險）
    cfg.pop("arch", None)

    # ✅ 只保留 MultiLayerWNN.__init__ 真的接受的參數
    sig = inspect.signature(MultiLayerWNN.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    cfg = {k: v for k, v in cfg.items() if k in allowed}

    model = MultiLayerWNN(**cfg).to(device)
    return model

def build_backbone_from_ckpt(path: str, device) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    """
    Reads ckpt saved by save_ckpt_v2 and returns backbone only.
    Return: model, backbone_cfg, extra
    """
    ckpt = torch.load(path, map_location=device)

    if "backbone_cfg" not in ckpt or "model_state_dict" not in ckpt:
        raise ValueError("ckpt missing backbone_cfg or model_state_dict")

    bb_cfg = ckpt["backbone_cfg"]
    model = build_backbone_only(bb_cfg, device)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print("[build_backbone_from_ckpt] missing:", missing)
    print("[build_backbone_from_ckpt] unexpected:", unexpected)

    return model, bb_cfg, ckpt.get("extra", {})