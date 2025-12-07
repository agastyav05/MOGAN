import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

# architecture: https://github.com/AKASHKADEL/dcgan-mnist/blob/master/main.py

class GeneratorNet(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32*2), nn.ReLU(True),
            nn.Conv2d(32*2, 32*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32*4), nn.ReLU(True),
            nn.Conv2d(32*4, z_dim, 4, 1, 0, bias=False),
            nn.Flatten()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 32*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(32*4), nn.ReLU(True),
            nn.ConvTranspose2d(32*4, 32*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32*2), nn.ReLU(True),
            nn.ConvTranspose2d(32*2, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        z = self.encoder(x)
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.decoder(z)
        return out

class DiscriminatorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 32 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(32 * 2, 32 * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(32 * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(), nn.Flatten(1)
        )

    def forward(self, x):
        return self.net(x)
    
criterion = nn.BCELoss()

def g_loss(x_fake, d_net):
    return criterion(d_net(x_fake), torch.ones(x_fake.size(0), device=x_fake.device).unsqueeze(1))

def d_loss(x_real, x_fake, d_net):
    return (criterion(d_net(x_real), torch.ones(x_real.size(0), device=x_fake.device).unsqueeze(1)) + criterion(d_net(x_fake), torch.zeros(x_real.size(0), device=x_fake.device).unsqueeze(1))) / 2

def train(batch_size=128, epochs=100, lr_g=2e-4, lr_d=2e-4, k=1, z_dim=100, seed=42):
    torch.manual_seed(seed)
    X = datasets.CIFAR10(root="data", download=True, transform=ToTensor())
    dl = DataLoader(X, batch_size=batch_size, shuffle=True, drop_last=True)

    g_net = GeneratorNet(z_dim=z_dim).to(device)
    d_net = DiscriminatorNet().to(device)
    opt_g = torch.optim.Adam(g_net.parameters(), lr=lr_g)
    opt_d = torch.optim.Adam(d_net.parameters(), lr=lr_d)

    # def sample_fake(x):
    #     mask = torch.zeros_like(x[:, :1])
    #     mask[:, :, 8:24, 8:24] = 1
    #     x = x * (1 - mask)
    #     x_fake = g_net(x, mask)
    #     return mask * x_fake + (1 - mask) * x

    def sample_fake(x):
        batch, _, height, width = x.size()
        mask = torch.zeros_like(x[:, :1])
        for i in range(batch):
            num_blocks = torch.randint(3, 7, (1,)).item()
            for _ in range(num_blocks):
                block_size = torch.randint(4, 9, (1,)).item()
                y = torch.randint(0, height - block_size, (1,)).item()
                z = torch.randint(0, width - block_size, (1,)).item()
                mask[i, :, y:y + block_size, z:z + block_size] = 1
        x_masked = x * (1 - mask)
        x_fake = g_net(x_masked, mask)
        x_final = mask * x_fake + (1 - mask) * x
        return x_final

    losses_g, losses_d = [], []
    running_loss_g, running_loss_d = 0, 0
    last_epoch = 0
    for epoch in range(1, epochs + 1):
        g_net.train()
        d_net.train()
        epoch_loss_g = 0
        epoch_loss_d = 0
        for x, _ in dl:
            x = x.to(device)
            m = batch_size
            for _ in range(k):
                x_fake = sample_fake(x)
                loss_d = d_loss(x, x_fake, d_net)
                epoch_loss_d += loss_d.item()
                opt_d.zero_grad()
                loss_d.backward()
                running_loss_d += loss_d.item()
                opt_d.step()
            x_fake = sample_fake(x)
            loss_g = g_loss(x_fake, d_net)
            epoch_loss_g += loss_g.item()
            opt_g.zero_grad()
            loss_g.backward()
            running_loss_g += loss_g.item()
            opt_g.step()
        losses_g.append(epoch_loss_g)
        losses_d.append(epoch_loss_d)
        if epoch % 1 == 0:
            print(f'[{epoch}] Generator loss: {running_loss_g / (epoch - last_epoch)}')
            print(f'[{epoch}] Discriminator loss: {running_loss_d / (epoch - last_epoch) / k}')
            running_loss_g = 0
            running_loss_d = 0
            last_epoch = epoch
        if epoch == epochs:
            plt.figure(figsize=(15, 10))
            plt.plot(range(1, epoch + 1), losses_g, range(1, epoch + 1), losses_d)
            plt.tight_layout()
            plt.savefig('loss.png')
    print('Training complete')
    return g_net, d_net

if __name__ == '__main__':
    print(f'Ready to begin training with {device}')
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()
    epochs = args.epochs if args.epochs else 100
    g_net, d_net = train(epochs=epochs)
    torch.save({
        "g_net_state_dict": g_net.state_dict(),
        "d_net_state_dict": d_net.state_dict()
    }, "models.pt")
    print('Saved models to models.pt')