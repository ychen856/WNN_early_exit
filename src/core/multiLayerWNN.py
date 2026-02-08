import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.wnnLutLayer import WNNLUTLayer

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

        layers = []
        prev_bits = in_bits

        self.layer_in_bits = []   # input bits per layer
        self.layer_out_luts = []  # number of LUTs per layer

        # for early exit start
        self.exit1_classifier = nn.Linear(hidden_luts[0], num_classes, bias=False)
        self.exit_tau = exit_tau  # 或者單獨設
        # for early exit end

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
        self.register_buffer("keep_idx", None)

    def forward(self, x_bits, return_hidden: bool = False):
        h = x_bits
        for layer in self.layers:
            h = layer(h)
        
        if self.keep_idx is not None:
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
        if self.keep_idx is not None:
            h_used = h[:, self.keep_idx]
        else:
            h_used = h
        logits = self.classifier(h_used) / self.tau
        return logits, h_list
    
    # for early exit start
    def forward_with_all_hidden_and_exits(self, x_bits: torch.Tensor):
        h_list = []
        h = x_bits
        exit1_logits = None

        for li, layer in enumerate(self.layers):
            h = layer(h)
            h_list.append(h)

            if li == 0:
                # --- select dims ---
                if hasattr(self, "exit1_keep_idx") and self.exit1_keep_idx is not None:
                    h_exit = h[:, self.exit1_keep_idx]
                else:
                    h_exit = h

                # --- normalize if available ---
                if hasattr(self, "exit1_mu") and hasattr(self, "exit1_sigma") and (self.exit1_mu is not None) and (self.exit1_sigma is not None):
                    h_exit = (h_exit - self.exit1_mu) / self.exit1_sigma

                # --- exit logits ---
                exit1_logits = self.exit1_classifier(h_exit) / self.exit_tau

        # final logits
        if self.keep_idx is not None:
            h_used = h[:, self.keep_idx]
        else:
            h_used = h
        final_logits = self.classifier(h_used) / self.tau

        return final_logits, exit1_logits, h_list
