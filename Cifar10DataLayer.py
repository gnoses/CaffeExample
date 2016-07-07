import skimage.io
import caffe
from tools import SimpleTransformer
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

class Cifar10DataLayerSync(caffe.Layer):
    def setup(self, bottom, top):
        # print 'Cifar10DataLayer.setup'
        self.top_names = ['data', 'label']
        params = eval(self.param_str)
        # check_params(params)
        # store input as class variables
        self.batch_size = params['batch_size']

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(params, None)

        top[0].reshape(
            self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])
        # Note the 20 channels (because PASCAL has 20 classes.)
        # top[1].reshape(self.batch_size, 10)
        top[1].reshape(self.batch_size)

    def forward(self, bottom, top):
        # Use the batch loader to load the next image.
        for i in range(self.batch_size):
            im, label = self.batch_loader.load_next_image()

            # Add directly to the caffe data layer
            top[0].data[i, ...] = im
            # print label.shape
            top[1].data[i, ...] = label

    def reshape(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):

        pass

class BatchLoader(object):

    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, params, result):
        self.result = result
        self.batch_size = params['batch_size']
        self.dataRoot = params['dataRoot']
        self.im_shape = params['im_shape']
        self._cur = 0  # current image
        # this class does some simple data-manipulations
        self.transformer = SimpleTransformer()

        # load cifar10 images into memory
        # 32 x 32 x 3 channel images
        imageLen = self.im_shape[0] * self.im_shape[1] * 3
        self.data = np.empty(shape=[0,imageLen])

        self.labelDense = []
        for i in range(1,6):
            data = self.LoadCifar10(self.dataRoot + '/data_batch_%d' % i)
            self.data = np.concatenate((self.data, data['data']), axis=0)
            self.labelDense += data['labels']
            # self.label = np.concatenate((self.label, npLabel), axis=0)
        self.dataCount = self.data.shape[0]
        # self.labelOneHot = np.zeros(shape=[self.dataCount, 10])
        self.labelOneHot = self.DenseToOneHot(np.array(self.labelDense), 10)
        # print self.labelOneHot
        self.indexList = range(self.dataCount) # shuffle()

        # print 'Batchloader initialized with %d images' % (self.dataCount)
        # print self.indexList

    # label_dense : m x 1
    # output : [m x num_classes] one hot
    def DenseToOneHot(self, labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        labelOneHot = np.zeros((labels_dense.shape[0], num_classes))
        labelOneHot[np.arange(labels_dense.shape[0]), np.int8(labels_dense)] = 1
        return np.int8(labelOneHot)

    def LoadCifar10(self, file):
        import cPickle
        with open(file, 'rb') as fo:
            dict = cPickle.load(fo)
        return dict

    def load_next_image(self):
        """
        Load the next image in a batch.
        """
        # Did we finish an epoch?
        if self._cur == self.dataCount:

            self._cur = 0
            shuffle(self.indexList)

        index = self.indexList[self._cur]  # Get the image index
        im = self.data[index,:].reshape(self.im_shape[2],self.im_shape[0],self.im_shape[1]).transpose([1,2,0])

        # plt.ioff()
        # plt.imshow(im)
        # plt.title(self.labelDense[index])
        # plt.show()

        # im = scipy.misc.imresize(im, self.im_shape)  # resize

        # do a simple horizontal flip as data augmentation
        # flip = np.random.choice(2)*2-1
        # im = im[:, ::flip, :]

        # Load and prepare ground truth
        # multilabel = np.zeros(20).astype(np.float32)

        self._cur += 1
        # return self.transformer.preprocess(im), self.labelOneHot[index,:]

        return self.transformer.preprocess(im), self.labelDense[index]
