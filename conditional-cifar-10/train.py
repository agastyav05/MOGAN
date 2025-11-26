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
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim + 10, 32*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(32*4), nn.ReLU(True),
            nn.ConvTranspose2d(32*4, 32*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32*2), nn.ReLU(True),
            nn.ConvTranspose2d(32*2, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, z, y):
        y = F.one_hot(y, num_classes=10).float().view(-1, 10, 1, 1)
        return self.net(torch.cat((z, y), 1))

class DiscriminatorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(13, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 32 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(32 * 2, 32 * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(32 * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(), nn.Flatten(1)
        )

    def forward(self, x, y):
        y = F.one_hot(y, num_classes=10).float().view(-1, 10, 1, 1)
        y = y.expand(-1, 10, x.size(2), x.size(3))
        return self.net(torch.cat([x, y], dim=1)).flatten(1)
    
criterion = nn.BCELoss()

def g_loss(x_fake, y_fake, d_net):
    return criterion(d_net(x_fake, y_fake), torch.ones(x_fake.size(0), device=x_fake.device).unsqueeze(1))

def d_loss(x_real, y_real, x_fake, y_fake, d_net):
    return (criterion(d_net(x_real, y_real), torch.ones(x_real.size(0), device=x_fake.device).unsqueeze(1)) + criterion(d_net(x_fake, y_fake), torch.zeros(x_real.size(0), device=x_fake.device).unsqueeze(1))) / 2

def train(batch_size=128, epochs=100, lr_g=2e-4, lr_d=2e-4, k=1, z_dim=100, seed=42):
    torch.manual_seed(seed)
    X = datasets.CIFAR10(root="data", download=True, transform=ToTensor())
    dl = DataLoader(X, batch_size=batch_size, shuffle=True, drop_last=True)

    g_net = GeneratorNet(z_dim=z_dim).to(device)
    d_net = DiscriminatorNet().to(device)
    opt_g = torch.optim.Adam(g_net.parameters(), lr=lr_g)
    opt_d = torch.optim.Adam(d_net.parameters(), lr=lr_d)

    def sample_fake(m):
        z = torch.randn(m, z_dim, 1, 1, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)
        return g_net(z, y), y

    losses_g, losses_d = [], []
    running_loss_g, running_loss_d = 0, 0
    last_epoch = 0
    for epoch in range(1, epochs + 1):
        g_net.train()
        d_net.train()
        epoch_loss_g = 0
        epoch_loss_d = 0
        for (x, y) in dl:
            x = x.to(device)
            y = y.to(device)
            m = batch_size
            for _ in range(k):
                x_fake, y_fake = sample_fake(m)
                loss_d = d_loss(x, y, x_fake, y_fake, d_net)
                epoch_loss_d += loss_d.item()
                opt_d.zero_grad()
                loss_d.backward()
                running_loss_d += loss_d.item()
                opt_d.step()
            x_fake, y_fake = sample_fake(m)
            loss_g = g_loss(x_fake, y_fake, d_net)
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