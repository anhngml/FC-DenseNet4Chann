import numpy as np
import os
import time
from PIL import Image
from skimage import io
from matplotlib import pyplot as plt
import errno
import sys

from dataset_loaders.parallel_loader import ThreadedDataset


floatX = 'float16'


class VaihingenDataset(ThreadedDataset):
    '''The CamVid motion based segmentation dataset

    The Cambridge-driving Labeled Video Database (CamVid) [Camvid1]_ provides
    high-quality videos acquired at 30 Hz with the corresponding
    semantically labeled masks at 1 Hz and in part, 15 Hz. The ground
    truth labels associate each pixel with one of 32 semantic classes.

    This loader is intended for the SegNet version of the CamVid dataset,
    that resizes the original data to 360 by 480 resolution and remaps
    the ground truth to a subset of 11 semantic classes, plus a void
    class.

    The dataset should be downloaded from [Camvid2]_ into the
    `shared_path` (that should be specified in the config.ini according
    to the instructions in ../README.md).

    Parameters
    ----------
    which_set: string
        A string in ['train', 'val', 'valid', 'test'], corresponding to
        the set to be returned.

    References
    ----------
    .. [Camvid1] http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
    .. [Camvid2]
       https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid
    '''
    name = 'Vaihingen'
    non_void_nclasses = 6
    _void_labels = [6] #[6]

    # optional arguments
    data_shape = (224, 224, 4)
    # mean = [0.39068785, 0.40521392, 0.41434407]
    # std = [0.29652068, 0.30514979, 0.30080369]

    _cmap = {
        0: (0, 0, 255),        # Building
        1: (0, 255, 0),        # Tree
        2: (0, 255, 255),      # Low vegetation
        3: (255, 255, 0),      # Car
        4: (255, 0, 0),        # Clutter/background
        5: (255, 255, 255),    # Impervious surfaces
        6: (0, 0, 0)           # Void
    }
    _mask_labels = {0: 'Building', 1: 'Tree', 2: 'Low_vegetation', 3: 'Car',
                    4: 'Clutter_background', 5: 'Impervious_surfaces', 6: 'Void'}

    _filenames = None
    _prefix_list = None

    @property
    def get_n_classes(self):
        return self.non_void_nclasses

    @property
    def get_void_labels(self):
        return self._void_labels

    @property
    def get_n_batches(self):
        return self.nbatches

    @property
    def get_n_samples(selfs):
        return selfs.nsamples

    @property
    def prefix_list(self):
        if self._prefix_list is None:
            # Create a list of prefix out of the number of requested videos
            self._prefix_list = np.unique(np.array([el[:6]
                                                    for el in self.filenames]))

        return self._prefix_list

    @property
    def filenames(self):
        if self._filenames is None:
            # Get file names for this set and year
            filenames = []
            with open(os.path.join(self.path, self.which_set + '.txt')) as f:
                for fi in f.readlines():
                    raw_name = fi.strip()
                    raw_name = raw_name.split("/")[4]
                    raw_name = raw_name.strip()
                    filenames.append(raw_name)
            self._filenames = filenames
        return self._filenames

    def __init__(self, which_set='train', *args, **kwargs):

        self.which_set = "val" if which_set == "valid" else which_set
        if self.which_set == "train":
            self.image_path = os.path.join(self.path, "train")
            self.mask_path = os.path.join(self.path, "trainannot")
        elif self.which_set == "val":
            self.image_path = os.path.join(self.path, "val")
            self.mask_path = os.path.join(self.path, "valannot")
        elif self.which_set == "test":
            self.image_path = os.path.join(self.path, "test")
            self.mask_path = os.path.join(self.path, "testannot")
        elif self.which_set == 'trainval':
            self.image_path = os.path.join(self.path, "trainval")
            self.mask_path = os.path.join(self.path, "trainvalannot")

        # constructing the ThreadedDataset
        # it also creates/copies the dataset in self.path if not already there
        super(VaihingenDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset."""
        per_subset_names = {}
        # Populate self.filenames and self.prefix_list
        filenames = self.filenames
        prefix_list = self.prefix_list

        # cycle through the different videos
        for prefix in prefix_list:
            per_subset_names[prefix] = [el for el in filenames if
                                        el.startswith(prefix)]
        return per_subset_names

    def load_sequence(self, sequence):
        """Load a sequence of images/frames

        Auxiliary function that loads a sequence of frames with
        the corresponding ground truth and their filenames.
        Returns a dict with the images in [0, 1], their corresponding
        labels, their subset (i.e. category, clip, prefix) and their
        filenames.
        """
        # from skimage import io
        X = []
        Y = []
        F = []

        for prefix, frame in sequence:
            # img = io.imread(os.path.join(self.image_path, frame))
            # mask = io.imread(os.path.join(self.mask_path, frame))

            img = Image.open(os.path.join(self.image_path, frame))
            mask = Image.open(os.path.join(self.mask_path, frame))

            img = np.asarray(img, dtype=np.uint8)
            img = np.transpose(img, (1, 0, 2))

            mask = np.asarray(mask, dtype=np.uint8)
            mask = mask.T

            img = img.astype(floatX) / 255.
            mask = mask.astype('int32')

            X.append(img)
            Y.append(mask)
            F.append(frame)

        arr_X = np.array(X)
        arr_Y = np.array(Y)
        arr_F = np.array(F)

        ret = {'data': arr_X, 'labels': arr_Y, 'subset': prefix, 'filenames': arr_F}
        return ret


def test():
    trainiter = VaihingenDataset(
        which_set='train',
        batch_size=10,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (224, 224)},
        return_one_hot=False,
        return_01c=True,
        return_list=True,
        use_threads=True)

    validiter = VaihingenDataset(
        which_set='valid',
        batch_size=5,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (224, 224)},
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=False)

    train_nsamples = trainiter.nsamples
    nbatches = trainiter.nbatches
    print("Train %d" % (train_nsamples))

    valid_nsamples = validiter.nsamples
    print("Valid %d" % (valid_nsamples))

    # Simulate training
    max_epochs = 2
    start_training = time.time()
    for epoch in range(max_epochs):
        start_epoch = time.time()
        for mb in range(nbatches):
            start_batch = time.time()

            val = trainiter.next()
            img = val[0][0]
            mask = val[1][0]
            print(img.shape)

            result = Image.new("RGB", (448, 224), 'black')

            offset = (0, 0, 224, 224)
            result.paste(Image.fromarray(img, 'RGB'), offset)
            offset = (224, 0, 448, 224)
            result.paste(Image.fromarray(mask, 'RGB'), offset)

            # result = np.asarray(result, dtype=np.uint8)
            # result = np.transpose(result, (1, 0, 2))
            # result.show()

            io.imshow(img.astype('float32'))
            plt.show()

            io.imshow((mask * 51).astype('float32'))
            plt.show()

            # sys.exit(errno.EACCES)
            # print(trainiter.next()[0].shape)

            print("Minibatch {}: {} seg".format(mb, (time.time() -
                                                     start_batch)))
        print("Epoch time: %s" % str(time.time() - start_epoch))
    print("Training time: %s" % str(time.time() - start_training))


def run_tests():
    test()


if __name__ == '__main__':
    run_tests()
