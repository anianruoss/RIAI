import time
from os import path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

PROJECT_ROOT = path.dirname(path.dirname(path.abspath(__file__)))

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

# PGD parameters
steps = 5
eps = 0.08
eps_step = 0.05


def fgsm(model, x, target, eps, targeted=True, clip_min=None, clip_max=None):
    input = x.clone().detach_().to(device)
    input.requires_grad_()
    target = torch.LongTensor([target]).to(device)

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

    # generate random point in +-eps box around x
    x = 2. * eps * torch.rand_like(x) - eps

    for i in range(k):
        # FGSM step
        x = fgsm(model, x, target, eps_step, targeted)
        # projection step
        x = torch.max(x_min.to(device), x)
        x = torch.min(x_max.to(device), x)

    if (clip_min is not None) or (clip_max is not None):
        x.clamp_(min=clip_min, max=clip_max)

    return x


def batched_pgd(model, x_batch, y_batch, k, eps, eps_step, targeted=True,
                clip_min=None, clip_max=None):
    n = x_batch.size()[0]
    xprime_batch_list = []

    for i in range(n):
        x = x_batch[i, ...]
        y = y_batch[i]
        xprime = pgd(
            model, x, y, k, eps, eps_step, targeted, clip_min, clip_max
        )
        xprime_batch_list.append(xprime)

    xprime_batch_tensor = torch.stack(xprime_batch_list)
    assert x_batch.size() == xprime_batch_tensor.size()

    return xprime_batch_tensor


def plot_loss_histogram(model, loss, x, y, batch_idx):
    """
    reproduce results from Madry et al. (arXiv:1706.06083)
    """
    model.eval()

    num_samples = 1000
    perturbed_losses = np.empty(num_samples)

    for i in range(num_samples):
        perturbed_x = pgd(
            model, x, y, 2 * steps, eps, eps_step, targeted=False
        ).to(device)
        perturbed_losses[i] = loss(
            model(perturbed_x), y.to(device)
        )

    plt.hist(
        perturbed_losses, label=f'batch: {batch_idx}',
        weights=np.zeros(num_samples) + 1. / num_samples
    )
    plt.xlim(left=0, right=4)
    plt.yscale('log')
    plt.xlabel('Loss value')
    plt.ylabel('log(frequency)')
    plt.yticks([])


def train_model(model, num_epochs, enable_defense=False):
    learning_rate = 0.0001
    tot_steps = 0

    opt = optim.Adam(params=model.parameters(), lr=learning_rate)
    ce_loss = torch.nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        t1 = time.time()
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            if batch_idx == 0:
                x_plot = x_batch[0, :]
                y_plot = y_batch[0].unsqueeze(0)

            if batch_idx % 200 == 0:
                plot_loss_histogram(model, ce_loss, x_plot, y_plot, batch_idx)

            if enable_defense:
                model.eval()
                x_batch = batched_pgd(
                    model, x_batch, y_batch, steps, eps, eps_step,
                    targeted=False
                )

            model.train()
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            tot_steps += 1
            opt.zero_grad()
            out = model(x_batch)
            batch_loss = ce_loss(out, y_batch)
            batch_loss.backward()
            opt.step()

        plt.legend(loc=3)
        plt.title(f'Loss Distribution for Defense = {enable_defense}')
        plt.savefig(f'loss_distribution_defense_{enable_defense}.png')
        plt.close()

        tot_test, tot_acc = 0.0, 0.0

        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            out = model(x_batch)
            pred = torch.max(out, dim=1)[1]
            acc = pred.eq(y_batch).sum().item()
            tot_acc += acc
            tot_test += x_batch.size()[0]

        t2 = time.time()

        print(
            'Epoch %d: Accuracy %.5lf [%.2lf seconds]' % (
                epoch, tot_acc / tot_test, t2 - t1)
        )


for enable_defense in [False, True]:
    print(f'enable defense: {enable_defense}')

    model_name = 'model.pt' if not enable_defense else 'model_defense.pt'
    model = Net().to(device)

    if not path.isfile(model_name):
        train_model(model, 1, enable_defense)
        torch.save(model.state_dict(), model_name)

    else:
        model.load_state_dict(
            torch.load(model_name, map_location=lambda storage, loc: storage)
        )

    model.eval()

    eval_size = 1000
    eval_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=eval_size, shuffle=False
    )
    images, labels = next(iter(eval_loader))
    images, labels = images.to(device), labels.to(device)
    print(
        'accuracy on unperturbed images:',
        torch.max(model(images), 1)[1].eq(labels).sum().item() / eval_size
    )
    images = batched_pgd(
        model, images, labels, steps, eps, eps_step, targeted=False
    )
    print(
        'accuracy on perturbed images:',
        torch.max(model(images), 1)[1].eq(labels).sum().item() / eval_size
    )
