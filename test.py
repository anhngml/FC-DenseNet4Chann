import sys
import numpy as np
import imp
import time
import theano
from lasagne.layers import get_output

from data_loader import load_data
from metrics import numpy_metrics, theano_metrics

from skimage import io
from matplotlib import pyplot as plt

from PIL import Image
import matplotlib
from matplotlib import pyplot as plt

# matplotlib.use('TkAgg')
# theano.config.optimizer = "None"


def test(config_path, weight_path):
    """
    This function builds the model defined in config_path and restores the weights defined in weight_path. It then
    reports the jaccard and global accuracy metrics on the CamVid test set.
    """

    cf = imp.load_source('cf', config_path)

    ###############
    #  Load data  #
    ###############

    print('-' * 75)
    # Load config file

    # Load data
    print('Loading data')
    batch_size = 10
    _, _, iterator = load_data(cf.dataset, batch_size=batch_size)

    n_classes = iterator.get_n_classes
    _, n_rows, n_cols = iterator.data_shape
    void_labels = iterator.get_void_labels

    ###################
    #  Compile model  #
    ###################

    # Print summary
    net = cf.net
    net.restore(weight_path)

    # Compile test functions
    prediction = get_output(
        net.output_layer, deterministic=True, batch_norm_use_averages=False)
    metrics = theano_metrics(
        prediction, net.target_var, n_classes, void_labels)

    print('Compiling functions')
    start_time_compilation = time.time()
    f = theano.function([net.input_var, net.target_var],
                        metrics, mode=theano.Mode(optimizer="fast_compile"))
    g = theano.function([net.input_var], prediction)
    print('Compilation took {:.3f} seconds'.format(
        time.time() - start_time_compilation))

    ###################
    #     Main loop   #
    ###################

    n_batches = iterator.get_n_batches
    I_tot = np.zeros(n_classes)
    U_tot = np.zeros(n_classes)
    acc_tot = 0.
    n_imgs = 0
    for i in range(n_batches):
        next_batch = iterator.next()
        X, Y = next_batch['data'], next_batch['labels']
        I, U, acc = f(X, Y[:, None, :, :])
        pred = g(X)
        pred_img = np.reshape(np.argmax(pred, axis=1), (224, 224))
        pred_img = pred_img * 51

        io.imshow(pred_img)
        plt.show()

        I_tot += I
        U_tot += U
        acc_tot += acc * batch_size
        n_imgs += batch_size

        # # Progression bar ( < 74 characters)
        sys.stdout.write('\r[{}%]'.format(int(100. * (i + 1) / n_batches)))
        sys.stdout.flush()

    labels = ['sky', 'building', 'column_pole', 'road', 'sidewalk', 'tree', 'sign', 'fence', 'car', 'pedestrian',
              'byciclist']

    for label, jacc in zip(labels, I_tot / U_tot):
        print('{} :\t{:.4f}'.format(label, jacc))
    print('Mean Jaccard', np.mean(I_tot / U_tot))
    print('Global accuracy', acc_tot / n_imgs)

    # To visualize an image : np.reshape(np.argmax(g(X), axis = 1), (360, 480))
    # with g = theano.function([net.input_var], prediction)
    # plt.imshow(np.reshape(np.argmax(g(X), axis=1), (360, 480)), interpolation='nearest')
    # plt.show()


if __name__ == '__main__':
    config_path = 'config/FC-DenseNet103.py'
    weight_path = 'weights/model.npz'
    test(config_path, weight_path)
