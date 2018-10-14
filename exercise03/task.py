import sys
from os import path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

PROJECT_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from exercise03.model import ConvNet

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 64

np.random.seed(42)
torch.manual_seed(42)

# load the data set without normalization operation
test_dataset = datasets.MNIST(
    path.join(PROJECT_ROOT, 'mnist_data'),
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)


class Normalize(nn.Module):
    def forward(self, x):
        return (x - 0.1307) / 0.3081


# load the body of the neural net trained last time...
model = torch.load(
    path.join(PROJECT_ROOT, 'exercise03/model.net'), map_location='cpu'
)

# ... and add the data normalization as a first "layer" to the network allowing
# to search for adversarial examples to the real image, rather than to the
# normalized image
model = nn.Sequential(Normalize(), model)

# create a version of the model that outputs the class probabilities
model_to_prob = nn.Sequential(model, nn.Softmax())

# put the neural net into evaluation mode (this disables features like dropout)
model.eval()
model_to_prob.eval()


# define a show function for later
def show(original, adv, model_to_prob, title=None):
    p0 = model_to_prob(original).detach().numpy()
    p1 = model_to_prob(adv).detach().numpy()

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(original.detach().numpy().reshape(28, 28), cmap='gray_r')
    axarr[0].set_title("Original, class: " + str(p0.argmax()))
    axarr[1].imshow(adv.detach().numpy().reshape(28, 28), cmap='gray_r')
    axarr[1].set_title("Adversarial, class: " + str(p1.argmax()))
    f.suptitle(title)

    print(title)
    print("Class\t\tOrig\tAdv")
    for i in range(10):
        print("Class {}:\t{:.2f}\t{:.2f}".format(
            i, float(p0[:, i]), float(p1[:, i]))
        )


def fgsm(model, x, target, eps, targeted=True, clip_min=None, clip_max=None):
    input = x.clone().detach()
    input.requires_grad_()

    logits = model(input)
    model.zero_grad()
    loss = nn.CrossEntropyLoss()(logits, torch.LongTensor([target]))
    loss.backward()

    if targeted:
        out = input - eps * input.grad.sign()
    else:
        out = input + eps * input.grad.sign()

    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)

    return out


def fgsm_targeted(model, x, target, eps, **kwargs):
    return fgsm(model, x, target, eps, targeted=True, **kwargs)


def fgsm_untargeted(model, x, label, eps, **kwargs):
    return fgsm(model, x, label, eps, targeted=False, **kwargs)


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


def pgd_targeted(model, x, target, k, eps, eps_step, **kwargs):
    return pgd(model, x, target, k, eps, eps_step, targeted=True, **kwargs)


def pgd_untargeted(model, x, label, k, eps, eps_step, **kwargs):
    return pgd(model, x, label, k, eps, eps_step, targeted=False, **kwargs)


original = torch.unsqueeze(test_dataset[0][0], dim=0)

adv_fgsm_targeted = fgsm_targeted(
    model, original, 2, 0.3, clip_min=0., clip_max=1.
)
show(original, adv_fgsm_targeted, model_to_prob, title='FGSM Targeted')

adv_fgsm_untargeted = fgsm_untargeted(
    model, original, 7, 0.3, clip_min=0., clip_max=1.
)
show(original, adv_fgsm_untargeted, model_to_prob, title='FGSM Untargeted')

adv_pgd_targeted = pgd_targeted(
    model, original, 2, 40, 0.1, 0.01, clip_min=0., clip_max=1.
)
show(original, adv_pgd_targeted, model_to_prob, title='PGD Targeted')

adv_pgd_untargeted = pgd_untargeted(
    model, original, 7, 40, 0.1, 0.01, clip_min=0., clip_max=1.
)
show(original, adv_pgd_untargeted, model_to_prob, title='PGD Untargeted')
plt.show()
