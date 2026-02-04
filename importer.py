from pathlib import Path
import numpy as np
from os.path import join

import torch
from torch.utils.data import TensorDataset, DataLoader
from datasets.LoadDatasets import MnistDataloader
from src.dataio.encode import compute_dt_thresholds, dt_thermometer_encode


import numpy as np



def sigmoid(x):
    # A safer small sigmoid to avoid extreme overflow
    return 1.0 / (1.0 + np.exp(-x))


class WNNLayer:
    """
    Single WNN layer loaded from NPZ.

    Attributes:
      - conn_idx: [num_luts, k] (int32)
      - table:    [num_luts, 2^k] (float32), raw LUT values (before sigmoid)
      - k:        LUT input size
      - in_bits:  expected input length (for sanity check)
    """
    def __init__(self, conn_idx, table, lut_input_size, in_bits=None):
        self.conn_idx = conn_idx.astype(np.int32)
        self.table = table.astype(np.float32)
        self.k = int(lut_input_size)
        self.num_luts = self.conn_idx.shape[0]
        self.in_bits = in_bits

    def forward(self, x):
        """
        x: numpy array, shape [in_bits], float32
        Returns: numpy array, shape [num_luts], float32

        PyTorch behavior:
          - binarize: x_bits = (x > 0.5)
          - build idx = (((b0)*2 + b1)*2 + ...) style (MSB first)
          - lookup table[j, idx]
          - apply sigmoid
        """
        x = x.astype(np.float32, copy=False)

        if self.in_bits is not None and x.shape[0] != self.in_bits:
            raise ValueError(
                f"Input length {x.shape[0]} != expected in_bits {self.in_bits}"
            )

        out = np.empty(self.num_luts, dtype=np.float32)

        # per-LUT loop, bit-exact 模仿 PyTorch for-loop
        for j in range(self.num_luts):
            idx_val = 0
            for b in range(self.k):
                src_idx = self.conn_idx[j, b]
                val = x[src_idx]
                bit = 1 if val > 0.5 else 0      # (x_bits > 0.5).float()
                idx_val = idx_val * 2 + bit      # idx = idx*2 + bit  (MSB first)

            lut_raw = self.table[j, idx_val]     # raw table output
            out[j] = sigmoid(lut_raw)            # final layer output

        return out


class WNNFPGAImporter:
    """
    Loads a full multilayer WNN .npz file and provides
    forward()/predict() for software simulation (golden model).
    """
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.data = data

        self.num_layers = int(data["num_layers"])
        self.input_bits = int(data["input_bits"])
        self.num_classes = int(data["num_classes"])

        # classifier weight: [num_classes, H_last]
        self.classifier_weight = data["classifier_weight"].astype(np.float32)

        self.layers = []
        for l in range(self.num_layers):
            prefix = f"layer{l}_"

            in_bits = int(data[prefix + "in_bits"])
            num_luts = int(data[prefix + "num_luts"])
            k = int(data[prefix + "lut_input_size"])

            conn = data[prefix + "conn_idx"]

            # quantized or not
            if (prefix + "table_q") in data.files:
                table_q = data[prefix + "table_q"]
                scale = float(data[prefix + "table_scale"])
                table = table_q.astype(np.float32) * scale
            else:
                table = data[prefix + "table"]

            layer = WNNLayer(conn_idx=conn,
                             table=table,
                             lut_input_size=k,
                             in_bits=in_bits)
            self.layers.append(layer)

    def forward(self, bit_vector):
        """
        bit_vector: numpy array, shape [input_bits]
          - for layer 0, is usually 0/1 (dt_thermometer_encode output)
          - (>0.5) binarization
        Returns:
          logits: numpy array, shape [num_classes]
        """
        x = bit_vector.astype(np.float32, copy=False)

        if x.shape[0] != self.input_bits:
            raise ValueError(
                f"Input length {x.shape[0]} != expected input_bits {self.input_bits}"
            )

        for layer in self.layers:
            x = layer.forward(x)

        # classifier: [C, H_last] @ [H_last]
        logits = self.classifier_weight @ x
        return logits

    def predict(self, bit_vector):
        logits = self.forward(bit_vector)
        return int(np.argmax(logits))


# ----------------------------------------------------
# Example Usage
# ----------------------------------------------------
if __name__ == "__main__":
    importer = WNNFPGAImporter("/Users/yi-chunchen/workspace/Adaptive_WNN/model/wnn_pruned_br60_lr60.npz")

    print("Layers:", importer.num_layers)
    print("Input bits:", importer.input_bits)
    print("Num classes:", importer.num_classes)

    # random test vector (simulate FPGA input)
    dummy = np.random.randint(0, 2, size=importer.input_bits).astype(np.float32)

    logits = importer.forward(dummy)
    pred = importer.predict(dummy)

    print("logits:", logits)
    print("predicted class:", pred)

    # complete dataset evaluation sample
    input_path = '/workspace/Adaptive_WNN/datasets'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                       test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    # CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z = 32  # 16 / 32 / 64
    # oneshot: use the training dataset, calculate DT thresholds + normalization
    thresholds, xmin, xmax = compute_dt_thresholds(x_train, z=z)

    # Encode train / test
    x_test_bits   = dt_thermometer_encode(x_test.to(device), thresholds, xmin, xmax)
    in_bits = x_test_bits.size(1)

    val_ds   = TensorDataset(x_test_bits, y_test)
    test_loader   = DataLoader(val_ds, batch_size=512, shuffle=False)

    
    x_test_bits_cpu = x_test_bits.cpu()
    y_test_cpu = y_test.cpu()

    correct = 0
    total = x_test_bits_cpu.shape[0]

    for xb, yb in zip(x_test_bits_cpu, y_test_cpu):
        pred = importer.predict(xb.numpy())
        if pred == int(yb.item()):
            correct += 1
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")
