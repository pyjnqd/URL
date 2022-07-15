import tensorflow as tf
import io
import PIL.Image as Image
import numpy
path = "/mnt/hdd1/wuhao/mnist_cifar10_cifar100/records/mnist/0.tfrecords"


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
    print(dataset.cardinality())
    # pass every single feature through our mapping function
    dataset = dataset.map(
        parse_tfr_element
    )
    print(dataset.cardinality())
    return dataset


dataset_small = get_dataset_small(path)
print(dataset_small.__len__())
for sample in dataset_small.take(5):
    image  = sample[0]
    image = Image.open(io.BytesIO(image.numpy()))
    image.save('test.jpg')
    print(sample[1])
