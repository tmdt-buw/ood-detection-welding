import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(EncBlock, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module(
            "Linear", nn.Linear(in_features=in_features, out_features=out_features)
        )

    def forward(self, x):
        out = self.layers(x)
        return torch.cat([x, out], dim=1)
        # return out


class LeNet5(nn.Module):

    def __init__(self, seq_len: int = 200):  # formerly 1080 hardcoded
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(
                in_channels=2, out_channels=6, kernel_size=3, stride=1, padding="same"
            ),
            nn.Tanh(),
        )

        self.layer2 = nn.AvgPool1d(kernel_size=2)

        self.layer3 = nn.Sequential(
            nn.Conv1d(
                in_channels=6, out_channels=16, kernel_size=3, stride=1, padding="same"
            ),
            nn.Tanh(),
        )

        self.layer4 = nn.AvgPool1d(kernel_size=2)

        self.layer5 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=120,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            nn.Tanh(),
        )

        if seq_len == 200:
            in_features = 120 * 49
        elif seq_len == 200 * 10:
            in_features = 120 * 499
        else:
            in_features = 120 * (49 + seq_len // 5)

        self.layer6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=in_features, out_features=84
            ),  # Adjust in_features based on input size
            nn.Tanh(),
        )

        # New enc block
        self.enc = EncBlock(in_features=84, out_features=84)

        # No classification required here
        # self.layer7 = nn.Linear(in_features=84, out_features=n_classes)

        self.all_layers = [
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            self.layer6,
        ]

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)
        x = self.layer1(x)  # Conv1D
        x = self.layer2(x)  # AvgPool
        x = self.layer3(x)  # Conv1D
        x = self.layer4(x)  # AvgPool
        x = self.layer5(x)  # Conv1D
        x = self.layer6(x)  # Linear

        # ENCODING
        enc_dim = 84
        feat = self.enc(x)
        mu = feat[:, :enc_dim]
        logvar = feat[:, enc_dim:]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        feat = eps.mul(std * 0.001).add_(mu)

        return feat  # x


class Regressor(nn.Module):
    def __init__(
        self,
        indim=168,
        n_transformation_classes: int = 2,
        seq_len: int = 200,
        device: str | torch.device = "cpu",
    ):  # indim = [orig_img_avg_pooled_features, trans_img_avg_pooled_features] = 2*enc_dim from forward, num_classes is the number of possible transformations = 4}
        super(Regressor, self).__init__()

        self.num_classes = n_transformation_classes
        self.indim = indim
        self.seq_len = seq_len
        self.device = device

        fc1_outdim = 42

        self.lenet = LeNet5(seq_len=seq_len)

        self.fc_quality = nn.Linear(indim, 2)

        self.fc1 = nn.Linear(indim, fc1_outdim)
        self.fc2 = nn.Linear(fc1_outdim, n_transformation_classes)

        self.relu1 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.to(device)

    def forward(
        self, x1, x2
    ):  # shape of x1 and x2 should be 5D = [BS, C=3, No. of images=clip_total_frames, 224, 224]
        # print("x2 shape before: ", x2.shape)
        # shape von x1 und x2 hier erstmal: 32,1,198,2
        x1 = self.lenet(x1)
        x2 = self.lenet(x2)  # now the shape of x1 = x2 = BS X 512
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        return x

    def forward_with_quality(
        self, x1, x2
    ):  # shape of x1 and x2 should be 5D = [BS, C=3, No. of images=clip_total_frames, 224, 224]
        # shape von x1 und x2 hier erstmal: 32,1,198,2
        x1 = self.lenet(x1)
        x2 = self.lenet(x2)  # now the shape of x1 = x2 = BS X 512

        x = torch.cat((x1, x2), dim=1)

        x_quality = self.fc_quality(x)  # Vorhersage für Qualität (0 oder 1)
        x_quality = torch.sigmoid(x_quality)

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        return x, x_quality
        
    def save_checkpoint(self, filepath: str) -> None:
        """
        Save model checkpoint including weights and hyperparameters.
        
        Args:
            filepath: Path where the checkpoint will be saved
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'indim': self.indim,
            'n_transformation_classes': self.num_classes,
            'seq_len': self.seq_len,
            'device': str(self.device)
        }
        torch.save(checkpoint, filepath)

    @staticmethod
    def load_checkpoint(filepath: str, device: str | torch.device = "cpu") -> "Regressor":
        """
        Load model from a checkpoint file.
        
        Args:
            filepath: Path to the checkpoint file
            device: Device to load the model to
            
        Returns:
            Loaded Regressor model
        """
        checkpoint = torch.load(filepath, map_location=device)
        model = Regressor(
            indim=checkpoint['indim'],
            n_transformation_classes=checkpoint['n_transformation_classes'],
            seq_len=checkpoint['seq_len'],
            device=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
