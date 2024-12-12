import torch
import torch.nn as nn

# d_loss 接近于 log(2)≈0.693  log(2)≈0.693，表示判别器在真实样本和生成样本之间达到了平衡。

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Ensure input has correct shape: (batch_size, 1, 128, 128, 128)
        x = x.view(-1, 1, 128, 128, 128)
        x = self.layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x
