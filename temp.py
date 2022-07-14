import tensorflow as tf
import io
import PIL.Image as Image
import numpy
path = '/home/wuhao/data/meta_dataset/records/mscoco/58.tfrecords'


def parse_tfr_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    }

    content = tf.io.parse_single_example(element, feature_description)

    label = content['label']
    image = content['image']

    return (image, label)


def get_dataset_small(filename):
    # create the dataset
    dataset = tf.data.TFRecordDataset(filename)

    # pass every single feature through our mapping function
    dataset = dataset.map(
        parse_tfr_element
    )

    return dataset


# dataset_small = get_dataset_small(path)
#
# for sample in dataset_small.take(1):
#     image  = sample[0]
#     image = Image.open(io.BytesIO(image.numpy()))
#     image.save('test.jpg')
#     print(sample[1])
import numpy as np
import torch
t =  list(np.reshape([[j]*5 for j in range(5)], (1,-1)).squeeze()) + list(np.reshape([[j]*10 for j in range(5)], (1,-1)).squeeze())
t = torch.LongTensor(t)
print(t)