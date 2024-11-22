import torch

class EffectClassifier1(torch.nn.Module):
    def __init__(self, n_classes, batch_size=4, embed_dim=768):
        super(EffectClassifier1, self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Flatten()
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(128 * 1764, embed_dim),  # Adjust input size to match flattened output
            torch.nn.ReLU(),
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, embed_dim),
        )
        self.attn = torch.nn.MultiheadAttention(embed_dim * 2, num_heads=8, dropout=.1, batch_first=True)
        self.mlp_cls = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 2, embed_dim),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, n_classes),
        )
    def forward(self, x_wet, x_dry):
        x_wet = self.cnn(x_wet.unsqueeze(1))  # Adjust unsqueeze dimension
        x_dry = self.cnn(x_dry.unsqueeze(1))  # Adjust unsqueeze dimension
        x_wet = self.mlp(x_wet)
        x_dry = self.mlp(x_dry)
        x = torch.cat([x_wet, x_dry], dim=1)
        x, _ = self.attn(x, x, x)  # Unpack attn output
        x = self.mlp_cls(x)
        return x

class EffectClassifier2(torch.nn.Module):
    def __init__(self, n_classes, embed_dim=768):
        super(EffectClassifier2, self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Flatten()
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(128 * 3810, embed_dim),  # Adjust input size to match flattened output
            torch.nn.ReLU(),
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, embed_dim),
        )
        self.attn = torch.nn.MultiheadAttention(embed_dim * 2, num_heads=2, dropout=.1, batch_first=True)
        self.fc = torch.nn.Linear(embed_dim * 2, embed_dim)
        self.cls = torch.nn.Linear(embed_dim, n_classes)
    def forward(self, x_wet, x_dry):
        x_wet = self.cnn(x_wet.unsqueeze(1))  # Adjust unsqueeze dimension
        x_dry = self.cnn(x_dry.unsqueeze(1))  # Adjust unsqueeze dimension
        x_wet = self.mlp(x_wet)
        x_dry = self.mlp(x_dry)
        x = torch.cat([x_wet, x_dry], dim=1)
        x, _ = self.attn(x, x, x)  # Unpack attn output
        x = self.cls(self.fc(x))
        return x