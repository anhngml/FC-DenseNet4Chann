import numpy as np
import matplotlib.pyplot as plt
import math
import sys

with np.load('errors.npz') as data:
    history = data['metrics'][()]
    b = data['best_epoch']
    train_acuracy = history['train']['accuracy']
    val_acuracy = history['val']['accuracy']
    train_jaccard = history['train']['jaccard']
    val_jaccard = history['val']['jaccard']
    train_loss = history['train']['loss']
    val_loss = history['val']['loss']

    print(history['train']['loss'][b])
    print(history['train']['accuracy'][b])
    print(history['train']['jaccard'][b])

    print(history['val']['loss'][b])
    print(history['val']['accuracy'][b])
    print(history['val']['jaccard'][b])

    curr_ep = len(history['train']['loss'])
    print(b)
    print(curr_ep - 1)

    epochs = np.arange(curr_ep)
    n_bins = math.ceil(len(epochs) / 10)

    fig = plt.figure(figsize=(26, 15))
    fig.suptitle('FC-DenseNet Performance - Best Epoch: {}'.format(b),
                 fontsize=14, fontweight='bold')
    plt.subplots_adjust(left=.08, bottom=.08, right=.95,
                        top=.95, wspace=.2, hspace=.25)
    # acuracy plot =====================
    y_stack = np.row_stack((np.array(train_acuracy), np.array(val_acuracy)))

    ax1 = fig.add_subplot(2, 2, (1, 2))
    ax1.grid(which='both')

    ax1.plot(epochs, y_stack[0, :], label='train', color='c', marker='o')
    ax1.plot(epochs, y_stack[1, :], label='val', color='g', marker='o')
    ax1.legend(loc=2)

    plt.xticks(epochs)
    plt.xlabel('epochs')
    plt.ylabel(
        'accuracy (best: {0:.4f})'.format(history['val']['accuracy'][b]))
    ax1.xaxis.get_major_locator().set_params(nbins=n_bins)
    ax1.yaxis.get_major_locator().set_params(nbins=20)

    # loss plot =====================
    y_stack = np.row_stack((np.array(train_loss), np.array(val_loss)))

    ax1 = fig.add_subplot(2, 2, 3)
    ax1.grid(which='both')

    ax1.plot(epochs, y_stack[0, :], label='train', color='c', marker='o')
    ax1.plot(epochs, y_stack[1, :], label='val', color='g', marker='o')
    ax1.legend(loc=2)

    plt.xticks(epochs)
    plt.xlabel('epochs')
    # plt.ylabel('loss')
    plt.ylabel(
        'loss (best: {0:.4f})'.format(history['val']['loss'][b]))

    ax1.xaxis.get_major_locator().set_params(nbins=n_bins)

    # jaccard plot =====================
    y_stack = np.row_stack((np.array(train_jaccard), np.array(val_jaccard)))

    ax1 = fig.add_subplot(2, 2, 4)
    ax1.grid(which='both')

    ax1.plot(epochs, y_stack[0, :], label='train', color='c', marker='o')
    ax1.plot(epochs, y_stack[1, :], label='val', color='g', marker='o')
    ax1.legend(loc=2)
    ax1.xaxis.get_major_locator().set_params(nbins=n_bins)
    # ax1.yaxis.get_major_locator().set_params(nbins=10)

    plt.xticks(epochs)
    plt.xlabel('epochs')
    # plt.ylabel('mean jaccard')
    plt.ylabel(
        'mean jaccard (best: {0:.4f})'.format(history['val']['jaccard'][b]))

    ax1.xaxis.get_major_locator().set_params(nbins=n_bins)
    ax1.yaxis.get_major_locator().set_params(nbins=20)

    plt.savefig('plot.png')

    plt.show()
