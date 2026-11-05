import torch.nn as nn

ACTS = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}


def conv_block(c_in, c_out, act):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        ACTS[act](),
        nn.MaxPool2d(2),
    )


class MosquitoCNN(nn.Module):
    """4 conv blocks -> global average pooling -> dense(hidden) -> output (5 logits)."""

    def __init__(self, num_classes=5, widths=(16, 32, 64, 128), hidden=64,
                 dropout=0.3, activation="relu"):
        super().__init__()
        blocks, c_in = [], 3
        for w in widths:
            blocks.append(conv_block(c_in, w, activation))
            c_in = w
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_in, hidden),
            ACTS[activation](),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),  # raw logits; softmax is inside the loss
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)