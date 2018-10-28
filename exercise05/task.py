import sys
import time
from os import path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
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


def show(result_gd, result_gd_bounds, result_lbfgsb, title):
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
    axarr[0].imshow(gd_i.reshape(28, 28), cmap='gray_r')
    axarr[0].set_title(f'Gradient Descent (success={gd_s})')
    axarr[1].imshow(gdb_i.reshape(28, 28), cmap='gray_r')
    axarr[1].set_title(f'Gradient Descent w. Bounds (success={gdb_s})')
    axarr[2].imshow(lbfgsb_i.reshape(28, 28), cmap='gray_r')
    axarr[2].set_title(f'L-BFGS-B (success={lbfgsb_s})')

    f.suptitle(title)
    plt.show()


nine = test_dataset[12][0]
fig = plt.figure()
plt.imshow(nine.numpy().reshape((28, 28)), cmap='gray_r')
plt.title('Original Nine')
plt.show()


def setup_i(box=1, init_zero=False, **kwargs):
    """
    Create the Optimization target composed of a fixed part (equal to the
    original image) and an variable part that is optimized.
    """
    if not (box in [1, 2]):
        box = 1

    if box == 1:
        i_fix = nine[0, 0:16, :].clone()
        i_var = nine[0, 16:, :].clone()
    elif box == 2:
        i_fix = nine.clone()
        i_var = nine[0, 16:, 7:14].clone()

    if init_zero:
        i_var.zero_()

    i_var.requires_grad_()

    return i_fix, i_var


def le(a, b, square=False, **kwargs):
    """
    Encodes the loss function for "a <= b". If square is false d = |a - b| is
    used, else d = (a - b)^2.
    """
    if square:
        return torch.clamp((a - b).sign() * (a - b) * (a - b), min=0)
    else:
        return torch.clamp(a - b, min=0)


def lt(a, b, **kwargs):
    """
    Encodes the loss function for "a <= b".
    """
    eps = 10e-15
    return le(a + eps, b, **kwargs)


def get_i(i_fix, i_var, box=1, **kwargs):
    """
    Combines the two parts of the target variable into the target variable.
    """
    if box != 2:
        i = torch.cat((i_fix, i_var), 0)
    else:
        i = i_fix.clone()
        i[0, 16:, 7:14] = i_var

    i = i.reshape((1, 1, 28, 28))

    return i


def get_loss(i_fix, i_var, square=False, add_bounds=False, use_logits=True,
             **kwargs):
    """
    Calculate the loss for the given query.
    """
    loss = 0

    if use_logits:
        o1 = NN1_logits(get_i(i_fix, i_var, **kwargs))
        o2 = NN2_logits(get_i(i_fix, i_var, **kwargs))
    else:
        o1 = NN1(get_i(i_fix, i_var, **kwargs))
        o2 = NN2(get_i(i_fix, i_var, **kwargs))

    for k in range(10):
        if k == 9:
            pass
        loss += lt(o1[0, k], o1[0, 9], square=square)
    for k in range(10):
        if k == 8:
            pass
        loss += lt(o2[0, k], o2[0, 8], square=square)

    if add_bounds:
        i_var_flat = i_var.view(-1)
        for k in range(i_var_flat.size()[0]):
            loss += le(0, i_var_flat[k], square=square)
            loss += le(i_var_flat[k], 1, square=square)

    return loss


def solve_gd(setup_var_fn, get_var_fn, loss_fn, max_iter=100, **kwargs):
    t0 = time.time()

    target_fix, target_var = setup_var_fn(**kwargs)
    opt = optim.SGD(params=[target_var], lr=0.1)

    for k in range(max_iter):
        loss = loss_fn(target_fix, target_var, **kwargs)

        if loss == 0:
            break

        opt.zero_grad()
        loss.backward()
        opt.step()

    t1 = time.time()

    loss = loss.detach().numpy()
    solved = (loss == 0)

    return (
        solved,
        loss,
        get_var_fn(target_fix, target_var, **kwargs).detach().numpy(),
        t1 - t0,
        k
    )


def solve_lbfgsb(setup_var_fn, get_var_fn, loss_fn, **kwargs):
    t0 = time.time()

    target_fix, target_var = setup_var_fn(**kwargs)

    def get_loss_():
        return loss_fn(target_fix, target_var, **kwargs)

    def zero_grads():
        NN1_logits.zero_grad()
        NN2_logits.zero_grad()
        NN1.zero_grad()
        NN2.zero_grad()
        if target_var.grad is not None:
            target_var.grad.zero_()

    lbfgsb(target_var, 0, 1, get_loss_, zero_grads)

    t1 = time.time()

    loss = loss_fn(target_fix, target_var, **kwargs).detach().numpy()
    solved = (loss == 0)

    return (
        solved,
        loss,
        get_var_fn(target_fix, target_var, **kwargs).detach().numpy(),
        t1 - t0
    )


show(
    solve_gd(setup_i, get_i, get_loss),
    solve_gd(setup_i, get_i, get_loss, add_bounds=True),
    solve_lbfgsb(setup_i, get_i, get_loss),
    'Query with Logits (Original Image Initialization)'
)
show(
    solve_gd(setup_i, get_i, get_loss, use_logits=False),
    solve_gd(setup_i, get_i, get_loss, add_bounds=True, use_logits=False),
    solve_lbfgsb(setup_i, get_i, get_loss, use_logits=False),
    'Query without Logits (Original Image Initialization)'
)
show(
    solve_gd(setup_i, get_i, get_loss, init_zero=True),
    solve_gd(setup_i, get_i, get_loss, add_bounds=True, init_zero=True),
    solve_lbfgsb(setup_i, get_i, get_loss, init_zero=True),
    'Query with Logits (Zero Initialization)'
)
show(
    solve_gd(setup_i, get_i, get_loss, use_logits=False, init_zero=True),
    solve_gd(
        setup_i, get_i, get_loss, add_bounds=True, use_logits=False,
        init_zero=True
    ),
    solve_lbfgsb(setup_i, get_i, get_loss, use_logits=False, init_zero=True),
    'Query without Logits (Zero Initialization)'
)
show(
    solve_gd(setup_i, get_i, get_loss, square=True),
    solve_gd(setup_i, get_i, get_loss, add_bounds=True, square=True),
    solve_lbfgsb(setup_i, get_i, get_loss, square=True),
    'Query with Logits and Squared Loss (Original Image Initialization)'
)
show(
    solve_gd(setup_i, get_i, get_loss, box=2),
    solve_gd(setup_i, get_i, get_loss, add_bounds=True, box=2),
    solve_lbfgsb(setup_i, get_i, get_loss, box=2),
    'Query with Logits + Box Constraints (Original Image Initialization)'
)
