import sys
caffe_root = '../../'
sys.path.append(caffe_root + 'python')
import caffe
from caffe import layers as L, params as P

sys.path.append(caffe_root + "examles/pycaffe/layers") # the datalayers we will use are in this directory.
sys.path.append(caffe_root + "examles/pycaffe") # the tools file is in this folder
import tools
from TrainingPlot import *

# helper function for common structures
def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,weight_filler=dict(type='xavier'))
    return conv, L.ReLU(conv, in_place=True)


# another helper function
def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout,weight_filler=dict(type='xavier'))
    return fc, L.ReLU(fc, in_place=True)


# yet another helper function
def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


# main netspec wrapper
def CaffenetModel(data_layer_params):
    # setup the python data layer
    n = caffe.NetSpec()
    n.data, n.label = L.Python(module='Cifar10DataLayer', layer='Cifar10DataLayerSync',
                               ntop=2, param_str=str(data_layer_params))

    # the net itself
    n.conv1, n.relu1 = conv_relu(n.data, 3, 128)
    n.pool1 = max_pool(n.relu1, 2, stride=2)
    n.norm1 = L.BatchNorm(n.pool1)

    # n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 3, 256)
    n.pool2 = max_pool(n.relu2, 2, stride=2)
    n.norm1 = L.BatchNorm(n.pool2)

    # n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    # n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1)
    # n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2)
    # n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2)
    # n.pool5 = max_pool(n.relu5, 3, stride=2)
    # n.fc6, n.relu6 = fc_relu(n.pool5, 4096)

    n.fc6, n.relu6 = fc_relu(n.pool2, 1024)
    n.drop6 = L.Dropout(n.relu6, in_place=True)
    n.fc7, n.relu7 = fc_relu(n.drop6, 1024)
    n.drop7 = L.Dropout(n.relu7, in_place=True)
    n.score = L.InnerProduct(n.drop7, num_output=10,weight_filler=dict(type='xavier'))
    # n.loss = L.SigmoidCrossEntropyLoss(n.score, n.label)
    n.loss = L.SoftmaxWithLoss(n.score, n.label)
    n.accuracy = L.Accuracy(n.score, n.label)
    print n.label
    return str(n.to_proto())

def hamming_distance(gt, est):
    return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))

def check_accuracy(net, num_batches, batch_size = 128):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data
        ests = net.blobs['score'].data > 0
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            acc += hamming_distance(gt, est)
            print gt
            print net.blobs['score'].data
    return acc / (num_batches * batch_size)

caffe.set_mode_gpu()
caffe.set_device(0)

solverprototxt = tools.CaffeSolver(trainnet_prototxt_path = "trainnet.prototxt", testnet_prototxt_path = "valnet.prototxt")
solverprototxt.sp['test_initialization'] = "true"
solverprototxt.sp['test_interval'] = '100000'
solverprototxt.sp['display'] = "100000"
solverprototxt.sp['snapshot'] = "1000"
solverprototxt.sp['snapshot_prefix'] = '"snapshot/"' # string withing a string!
solverprototxt.sp['base_lr'] = "0.001"
solverprototxt.write('solver.prototxt')

dataRoot = '../../data/cifar10'
batchSize = 128

# write train net.
with open('trainnet.prototxt', 'w') as f:
    # provide parameters to the data layer as a python dictionary. Easy as pie!
    data_layer_params = dict(batch_size = batchSize, im_shape = [32, 32, 3], split = 'train', dataRoot = dataRoot)
    f.write(CaffenetModel(data_layer_params))

# write validation net.
with open('valnet.prototxt', 'w') as f:
    data_layer_params = dict(batch_size = batchSize, im_shape = [32, 32, 3], split = 'val', dataRoot = dataRoot)
    f.write(CaffenetModel(data_layer_params))

# solver = caffe.SGDSolver('solver.prototxt')
solver = caffe.AdamSolver('solver.prototxt')
# solver.net.copy_from(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
# solver.test_nets[0].share_with(solver.net)
trainingPlot = TrainingPlot()
iter = 1000
trainingPlot.SetConfig(batchSize, 50000, iter)

# exit(0)
for i in range(iter):
    solver.step(1)
    # print solver.net.blobs['loss'].data.shape
    trainLoss = solver.net.blobs['loss'].data.item(0)
    trainAcc = solver.net.blobs['accuracy'].data.item(0)
    # ests = solver.net.blobs['score'].data[0]
    # print ests

    if i % 100 == 0:
        solver.test_nets[0].forward()  # test net (there can be more than one)
        valLoss = solver.test_nets[0].blobs['loss'].data.item(0)
        valAcc = solver.test_nets[0].blobs['accuracy'].data.item(0)

    # print i , 'accuracy:{0:.4f}'.format(valAcc)
    trainingPlot.Add(i, trainLoss, valLoss, trainAcc, valAcc)
    trainingPlot.Show()
