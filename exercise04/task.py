import time
from os import path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms

PROJECT_ROOT = path.dirname(path.dirname(path.abspath(__file__)))

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 64

np.random.seed(42)
torch.manual_seed(42)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        # add data normalization to the network as a first "layer" allowing to
        # search for adversarial examples to the real image, rather than to the
        # normalized image
        x = (x - 0.1307) / 0.3081
        x = x.view((-1, 28 * 28))
        x = F.relu(self.fc(x))
        x = self.fc2(x)
        return x


train_dataset = datasets.MNIST(
    path.join(PROJECT_ROOT, 'mnist_data'),
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)
test_dataset = datasets.MNIST(
    path.join(PROJECT_ROOT, 'mnist_data'),
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

model = Net()
model = model.to(device)
model.train()


def fgsm(model, x, target, eps, targeted=True, clip_min=None, clip_max=None):
    input = x.clone().detach()
    input.requires_grad_()

    logits = model(input)
    model.zero_grad()
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()

    if targeted:
        out = input - eps * input.grad.sign()
    else:
        out = input + eps * input.grad.sign()

    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)

    return out


def pgd(model, x, target, k, eps, eps_step, targeted=True, clip_min=None,
        clip_max=None):
    x_min = x - eps
    x_max = x + eps

    for i in range(k):
        # FGSM step
        x = fgsm(model, x, target, eps_step, targeted)
        # projection step
        x = torch.max(x_min, x)
        x = torch.min(x_max, x)

    if (clip_min is not None) or (clip_max is not None):
        x.clamp_(min=clip_min, max=clip_max)

    return x


if not path.isfile('model.pt'):
    learning_rate = 0.0001
    num_epochs = 20

    opt = optim.Adam(params=model.parameters(), lr=learning_rate)

    ce_loss = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter()
    tot_steps = 0

    for epoch in range(1, num_epochs + 1):
        t1 = time.time()
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = pgd(
                model, x_batch, y_batch, 40, 0.1, 0.01, targeted=False,
                clip_min=0., clip_max=1.
            )

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            tot_steps += 1
            opt.zero_grad()
            out = model(x_batch)
            batch_loss = ce_loss(out, y_batch)

            if batch_idx % 100 == 0:
                pred = torch.max(out, dim=1)[1]
                acc = pred.eq(y_batch).sum().item() / float(batch_size)

                writer.add_scalar('data/accuracy', acc, tot_steps)
                writer.add_scalar('data/loss', batch_loss.item(), tot_steps)

            batch_loss.backward()
            opt.step()

        tot_test, tot_acc = 0.0, 0.0
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            out = model(x_batch)
            pred = torch.max(out, dim=1)[1]
            acc = pred.eq(y_batch).sum().item()
            tot_acc += acc
            tot_test += x_batch.size()[0]
        t2 = time.time()

        print('Epoch %d: Accuracy %.5lf [%.2lf seconds]' % (
            epoch, tot_acc / tot_test, t2 - t1))

    torch.save(model.state_dict(), 'model.pt')

else:
    model.load_state_dict(torch.load('model.pt'))

model.eval()
