import cv2
import sklearn
from sklearn.cluster import KMeans
import scipy.cluster.vq as vq
import numpy as np
import os
import struct


DSIFT_STEP_SIZE = 4

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def load_mnist(dataset = "training", path = "./data/raw"):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    return img.astype(np.uint8), lbl.astype(np.uint8)

def load_cifar10_data(dataset):
    if dataset == 'train':
        with open('./cifar10/train/train.txt','r') as f:
            paths = f.readlines()
    if dataset == 'test':
        with open('./cifar10/test/test.txt','r') as f:
            paths = f.readlines()
    x, y = [], []
    for each in paths:
        each = each.strip()
        path, label = each.split(' ')
        img = cv2.imread(path)
        x.append(img)
        y.append(label)
    return [x, y]

def load_my_data(path, test=None):
    with open(path, 'r') as f:
        paths = f.readlines()
    x, y = [], []
    for each in paths:
        each = each.strip()
        label, path = each.split(' ')
        img = cv2.imread(path)
        if img.shape[:2] != (256,256):
            img = cv2.resize(img, (256,256))
        x.append(img)
        y.append(label)
    return [x, y]

def extract_sift_descriptors(img):
    """
    Input BGR numpy array
    Return SIFT descriptors for an image
    Return None if there's no descriptor detected
    """
    gray = img
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors


def extract_DenseSift_descriptors(img):
    """
    Input BGR numpy array
    Return Dense SIFT descriptors for an image
    Return None if there's no descriptor detected
    """
    gray = img
    sift = cv2.xfeatures2d.SIFT_create()

    # opencv docs DenseFeatureDetector
    # opencv 2.x code
    #dense.setInt('initXyStep',8) # each step is 8 pixel
    #dense.setInt('initImgBound',8)
    #dense.setInt('initFeatureScale',16) # each grid is 16*16
    disft_step_size = DSIFT_STEP_SIZE
    keypoints = [cv2.KeyPoint(x, y, disft_step_size)
            for y in range(0, gray.shape[0], disft_step_size)
                for x in range(0, gray.shape[1], disft_step_size)]

    keypoints, descriptors = sift.compute(gray, keypoints)

    #keypoints, descriptors = sift.detectAndCompute(gray, None)
    return [keypoints, descriptors]


def build_codebook(X, voc_size):
    """
    Inupt a list of feature descriptors
    voc_size is the "K" in K-means, k is also called vocabulary size
    Return the codebook/dictionary
    """
    features = np.vstack((descriptor for descriptor in X))
    kmeans = KMeans(n_clusters=voc_size, n_jobs=-2)
    kmeans.fit(features)
    codebook = kmeans.cluster_centers_.squeeze()
    return codebook


def input_vector_encoder(feature, codebook):
    """
    Input all the local feature of the image
    Pooling (encoding) by codebook and return
    """
    code, _ = vq.vq(feature, codebook)
    word_hist, bin_edges = np.histogram(code, bins=range(codebook.shape[0] + 1), normed=True)
    return word_hist
