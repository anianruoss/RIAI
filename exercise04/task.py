import time
from os import makedirs, path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

PROJECT_ROOT = path.dirname(path.dirname(path.abspath(__file__)))

if not path.isdir(path.join(PROJECT_ROOT, 'exercise04/plots')):
    makedirs(path.join(PROJECT_ROOT, 'exercise04/plots'))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

# PGD parameters
eps = 0.1
eps_step = 0.01
steps = 40


def fgsm(model, x, target, eps, targeted=True, clip_min=None, clip_max=None):
    input = x.clone().detach().to(device)
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

    tot_steps = 0

    rand_train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True
    )

    for epoch in range(1, num_epochs + 1):
        t1 = time.time()
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = pgd(
                model, x_batch, y_batch, steps, eps, eps_step, targeted=False,
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

                # reproduce results from Madry et al. (arXiv:1706.06083)
                image, label = next(iter(rand_train_loader))
                x_min = image - eps
                x_max = image + eps

                perturbed_losses = np.empty(1000)

                for i in range(1000):
                    start = (x_min - x_max) * torch.rand(image.shape) + x_max
                    perturbed_image = pgd(
                        model, start, label, steps, eps, eps_step, clip_min=0.,
                        clip_max=1.
                    )
                    perturbed_losses[i] = ce_loss(model(perturbed_image), label)

                plt.hist(perturbed_losses, label=f'batch {batch_idx}')
                plt.yscale('log', nonposy='clip')
                plt.xlabel('Loss value')
                plt.ylabel('log(frequency)')
                plt.yticks([])

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

        plt.legend()
        plt.savefig(
            path.join(
                PROJECT_ROOT, 'exercise04/plots', f'frequency_epoch_{epoch}.png'
            )
        )
        plt.close()
        print('Epoch %d: Accuracy %.5lf [%.2lf seconds]' % (
            epoch, tot_acc / tot_test, t2 - t1))

    torch.save(model.state_dict(), 'model.pt')

else:
    model.load_state_dict(
        torch.load('model.pt', map_location=lambda storage, loc: storage)
    )

model.eval()

eval_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1000, shuffle=False
)
images, labels = next(iter(eval_loader))
print('accuracy on unperturbed images:',
      torch.max(model(images), 1)[1].eq(labels).sum().item() / labels.size()[0])
images = pgd(
    model, images, labels, steps, eps, eps_step, targeted=False, clip_min=0.,
    clip_max=1.
)
print('accuracy on perturbed images:',
      torch.max(model(images), 1)[1].eq(labels).sum().item() / labels.size()[0])
