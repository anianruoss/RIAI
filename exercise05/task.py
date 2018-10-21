import sys
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
sys.path.append(PROJECT_ROOT)

from exercise05.lbfgsb import lbfgsb
from exercise05.model import ConvNet, Net

device = torch.device('cpu')

np.random.seed(42)
torch.manual_seed(42)

test_dataset = datasets.MNIST(
    path.join(PROJECT_ROOT, 'mnist_data'),
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)


class Normalize(nn.Module):
    def forward(self, x):
        return (x - 0.1307) / 0.3081


NN1_logits = torch.load('model_conv.net', map_location='cpu')
NN2_logits = torch.load('model_ff.net', map_location='cpu')

NN1_logits = nn.Sequential(Normalize(), NN1_logits)
NN2_logits = nn.Sequential(Normalize(), NN2_logits)

NN1 = nn.Sequential(NN1_logits, nn.Softmax())
NN2 = nn.Sequential(NN2_logits, nn.Softmax())

NN1_logits.eval()
NN2_logits.eval()
NN1.eval()
NN2.eval()


def show(result_gd, result_gd_bounds, result_lbfgsb):
    gd_s, gd_l, gd_i, gd_t, gd_n = result_gd
    gdb_s, gdb_l, gdb_i, gdb_t, gdb_n = result_gd_bounds
    lbfgsb_s, lbfgsb_l, lbfgsb_i, lbfgsb_t = result_lbfgsb

    def print_res(title, solved, loss, i, time, it=None):
        print(title + ':')
        print('\tSolved:', solved)
        print('\tLoss:', loss)
        print('\tTime:', time, 's')
        if it is not None:
            print('\tGradient Descent iterations:', it)
        p1 = NN1(torch.from_numpy(i).reshape((1, 1, 28, 28))).detach().numpy()
        p2 = NN2(torch.from_numpy(i).reshape((1, 1, 28, 28))).detach().numpy()
        print('\tNN1_logits class: {} (p = {:.2f}) '.format(
            p1.argmax(), p1.max())
        )
        print('\tNN2_logits class: {} (p = {:.2f}) '.format(
            p2.argmax(), p2.max())
        )

    print_res('Gradient Descent', gd_s, gd_l, gd_i, gd_t, gd_n)
    print_res('Gradient Descent w. Bounds', gdb_s, gdb_l, gdb_i, gdb_t, gdb_n)
    print_res('L-BFGS-B', lbfgsb_s, lbfgsb_l, lbfgsb_i, lbfgsb_t)

    f, axarr = plt.subplots(1, 3, figsize=(18, 16))
    axarr[0].imshow(gd_i.reshape(28, 28), cmap='gray')
    axarr[0].set_title('Gradient Descent')
    axarr[1].imshow(gdb_i.reshape(28, 28), cmap='gray')
    axarr[1].set_title('Gradient Descent w. Bounds')
    axarr[2].imshow(lbfgsb_i.reshape(28, 28), cmap='gray')
    axarr[2].set_title('L-BFGS-B')


nine = test_dataset[12][0]
fig = plt.figure()
plt.imshow(nine.numpy().reshape((28, 28)), cmap='gray_r')
plt.title('Original Nine')


# - Create these two tensors once and then have a function that combines them
# and calculates the loss.

# - For the loss it is easiest to implement a function implements the loss
# translation for the less-or-equal ($\leq$) and less ($<$) operators from the
# lecture. You can express all parts of the loss function with this. Make
# this parametric in the choice of d.

# - If implemented correctly your code should not run more than a few seconds.

# - There is no L-BFGS-B optimizer for pytroch yet. We provide a function that
# uses scipy to do this instead (see below).

def query_loss(i_const, i_var, nn1, nn2):
    image = torch.cat((i_const, i_var)).unsqueeze(0).unsqueeze(1)

    output_nn1 = nn1(image)
    output_nn2 = nn2(image)

    loss = torch.zeros(1)

    for i in range(9):
        if i >= 7:
            i += 1
        loss += torch.max(torch.zeros(1), output_nn1[0, i] - output_nn1[0, 8])

    for i in range(9):
        if i >= 8:
            i += 1
        loss += torch.max(torch.zeros(1), output_nn2[0, i] - output_nn2[0, 8])

    return loss


def solve_gd(max_iter=100, use_logits=True, **kwargs):
    t0 = time.time()
    t1 = time.time()

    nn1 = NN1_logits if use_logits else NN1
    nn2 = NN2_logits if use_logits else NN2

    i_const = nine[0, 0:16, :]
    i_var = nine[0, 16:, :]
    i_var.requires_grad_(True)

    assert torch.equal(torch.cat((i_const, i_var)).unsqueeze(0), nine)

    for it in range(max_iter):
        loss = query_loss(i_const, i_var, nn1, nn2)
        loss.backward()
        i_var.data -= 0.001 * i_var.grad.data
        i_var.grad.data.zero_()

        image = torch.cat((i_const, i_var)).unsqueeze(0).unsqueeze(1)

        solved = torch.equal(
            torch.argmax(nn1(image).data, 1), torch.LongTensor([8])
        ) and torch.equal(
            torch.argmax(nn2(image).data, 1), torch.LongTensor([9])
        )

        if solved:
            break

    return (
        solved,
        loss.detach()[0],
        torch.cat((i_const, i_var)).detach().numpy(),
        t1 - t0,
        it + 1
    )


# feel free to add args to this function
def solve_lbfgsb(**kwargs):
    t0 = time.time()
    t1 = time.time()
    loss = 0
    solved = False

    # Hint:
    # Use the provided lbfgsb(var, min_val, max_val, loss_fn, zero_grad_fn)
    # function.
    # It takes the tensor to optimize (var), the min and max value for each entry (a scalar),
    # a function that returns the current loss-tensor and a function that sets the 
    # gradients of everything used (NN1_logits, NN2_logits) and i_var to zero.
    # This function does not return anything but changes var.

    # return:
    # solved: Bool; did you find a solution
    # loss: Float; value of loss at the end
    # i: numpy array; the resulting i
    # t: float; how long the execution took
    return solved, loss, nine.detach().numpy(), t1 - t0


# using logits, initialized with zeros
show(
    solve_gd(init_zero=True),
    solve_gd(add_bounds=True, init_zero=True),
    solve_lbfgsb(init_zero=True)
)

# using logits, initialized with original image
show(
    solve_gd(),
    solve_gd(add_bounds=True),
    solve_lbfgsb()
)

# using probabilities, initialized with zeros
show(
    solve_gd(use_logits=False, init_zero=True),
    solve_gd(use_logits=False, init_zero=True),
    solve_lbfgsb(use_logits=False, init_zero=True)
)

# using probabilities, initialized with original image
show(
    solve_gd(use_prob=True),
    solve_gd(add_bounds=True, use_prob=True),
    solve_lbfgsb(use_prob=True)
)

# We see that using probabilities is not a viable approach. The numerical
# optimization problem becomes basically intractable due to the softmax
# function.

# ## different box constraint (task 1.7; optional), using logits
show(
    solve_gd(box=2),
    solve_gd(add_bounds=True, box=2),
    solve_lbfgsb(box=2)
)

# Since the region covered by box 2 is mostly empty it does not matter much
# whether we use init_zero or not.
