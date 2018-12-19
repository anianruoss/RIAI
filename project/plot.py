import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(8, 2))

# box relu
ax[0].plot([-4, 0, 4], [0, 0, 4], linestyle='--', label='ReLU')
ax[0].set_xlabel('$x$')
ax[0].set_ylabel('$ReLU(x)$')
ax[0].set_title('ReLU with Interval Domain')
ax[0].hlines(4, -4, 4, colors='r', linestyles=':', label='Approximation')
ax[0].hlines(0, -4, 4, colors='r', linestyles=':')
ax[0].vlines(-4, 0, 4, colors='r', linestyles=':')
ax[0].vlines(4, 0, 4, colors='r', linestyles=':')
ax[0].legend()

# lp relu
ax[1].plot([-4, 0, 4], [0, 0, 4], linestyle='--', label='ReLU')
ax[1].set_xlabel('$x$')
ax[1].set_ylabel('$ReLU(x)$')
ax[1].set_title('ReLU with Linear Program')
ax[1].plot([0, 4], [0, 4], color='r', linestyle=':', label='Approximation')
ax[1].plot([-4, 4], [0, 4], color='r', linestyle=':')
ax[1].hlines(0, -4, 0, colors='r', linestyles=':')
ax[1].legend()

plt.savefig('relu.png', bbox_inches='tight')
