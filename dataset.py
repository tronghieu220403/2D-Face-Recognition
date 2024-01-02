import tensorflow as tf
import numpy as np

def preprocess_image(filename, target_shape=(224,224)):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image


def create_image_path_ds(txt_file, img_folder):
    txt_filename = txt_file.replace('\\','/').split('/')[-1]
    with open(txt_file, 'rt') as f:
        lines = f.readlines()
    anchor = []
    validation = [] 
    for line in lines:
        a,v = line.split(' ')
        anchor.append(img_folder+a.strip()+'.png')
        validation.append(img_folder+v.strip()+'.png')
    # ds_av = tf.data.Dataset.from_tensor_slices((anchor, validation))
    a = tf.data.Dataset.from_tensor_slices(anchor)
    v = tf.data.Dataset.from_tensor_slices(validation)
    
    if('same' in txt_filename):
        l = tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))
    elif('diff' in txt_filename):
        l = tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))
    ds = tf.data.Dataset.zip((a,v, l))
    return ds


def create_image_path_tripless_ds(txt_file, img_folder):
    txt_filename = txt_file.replace('\\','/').split('/')[-1]
    with open(txt_file, 'rt') as f:
        lines = f.readlines()
    anchor = []
    positive = [] 
    negative = []
    for line in lines:
        a,p,n = line.strip().split(' ')
        anchor.append(img_folder+a+'.png')
        positive.append(img_folder+p+'.png')
        negative.append(img_folder+n+'.png')

    # ds_av = tf.data.Dataset.from_tensor_slices((anchor, validation))
    a = tf.data.Dataset.from_tensor_slices(anchor)
    p = tf.data.Dataset.from_tensor_slices(positive)
    n = tf.data.Dataset.from_tensor_slices(negative)
    
    ds = tf.data.Dataset.zip((a, p, n))
    return ds


