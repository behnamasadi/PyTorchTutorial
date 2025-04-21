import gzip
import numpy as np
import matplotlib.pyplot as plt

# all data files can be downloaded from http://yann.lecun.com/exdb/mnist/
# The training set contains 60000 examples, and the test set 10000 examples.

root_dir_path='data'

test_images='t10k-images-idx3-ubyte.gz'
test_labels='t10k-labels-idx1-ubyte.gz'
train_images='train-images-idx3-ubyte.gz'
train_labels='train-labels-idx1-ubyte.gz'



def reading_images(image_path,image_size , num_images, skipping_bytes ):
    f = gzip.open(image_path, 'r')
    f.read(skipping_bytes)
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)
    return data

def reading_labels(label_path,num_labels,skipping_bytes):
    f = gzip.open(label_path, 'r')
    f.read(skipping_bytes)
    buf = f.read(num_labels)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels




image_path=root_dir_path + '/' + train_images
num_train_images = 60000
image_size = 28
skipping_bytes=16
# skipping non-image information
test_images_data=reading_images(image_path,image_size , num_train_images, skipping_bytes )
image = np.asarray(test_images_data[-3]).squeeze()
plt.imshow(image)
plt.show()


label_path=root_dir_path + '/' + train_labels
# skipping non-image information
skipping_bytes=8
num_labels= 60000
labels=reading_labels(label_path,num_labels,skipping_bytes)
print(labels[-3])
    

# Using idx2numpy
# import idx2numpy
# import numpy as np
# file = 'data/train-images-idx3-ubyte'
# arr = idx2numpy.convert_from_file(file)
# # arr is now a np.ndarray type of object of shape 60000, 28, 28