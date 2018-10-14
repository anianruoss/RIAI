import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 64

np.random.seed(42)
torch.manual_seed(42)

# loading the data set without normalization operation
test_dataset = datasets.MNIST('mnist_data/', train=False, download=True,
                              transform=transforms.Compose(
                                  [transforms.ToTensor()]
                              ))


class Normalize(nn.Module):
    def forward(self, x):
        return (x - 0.1307) / 0.3081


# load the body of the neural net trained last time...
model = torch.load('model.net', map_location='cpu')

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
def show(original, adv, model_to_prob):
    p0 = model_to_prob(original).detach().numpy()
    p1 = model_to_prob(adv).detach().numpy()
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(original.detach().numpy().reshape(28, 28), cmap='gray')
    axarr[0].set_title("Original, class: " + str(p0.argmax()))
    axarr[1].imshow(adv.detach().numpy().reshape(28, 28), cmap='gray')
    axarr[1].set_title("Original, class: " + str(p1.argmax()))
    print("Class\t\tOrig\tAdv")
    for i in range(10):
        print("Class {}:\t{:.2f}\t{:.2f}".format(i, float(p0[:, i]),
                                                 float(p1[:, i])))


def fgsm_targeted(model, x, target, eps):
    # TODO: implement
    return x


def fgsm_untargeted(model, x, label, eps):
    # TODO: implement
    return x


def pgd_targeted(model, x, target, k, eps, eps_step):
    # TODO: implement
    return x


def pgd_untargeted(model, x, label, k, eps, eps_step):
    # TODO: implement
    return x


# try out attacks
original = torch.unsqueeze(test_dataset[0][0], dim=0)
adv = pgd_untargeted(
    model, original, 7, 10, 0.08, 0.05, clip_min=0, clip_max=1.0
)
show(original, adv, model_to_prob)
plt.show()
