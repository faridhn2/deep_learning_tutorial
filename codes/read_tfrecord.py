import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2




data_path1 = 'F:/tfrecords/train1_19.tfrecords'
data_path2 = 'F:/tfrecords/train19_22.tfrecords'
data_path3 = 'F:/tfrecords/train22_25.tfrecords'
data_path4 = 'F:/tfrecords/train25_27.tfrecords'
data_path5 = 'F:/tfrecords/train27_29.tfrecords'
data_path6 = 'F:/tfrecords/train29_31.tfrecords'
data_path7 = 'F:/tfrecords/train31_33.tfrecords'
data_path8 = 'F:/tfrecords/train33_35.tfrecords'
data_path9 = 'F:/tfrecords/train35_37.tfrecords'
data_path10 = 'F:/tfrecords/train37_39.tfrecords'
data_path11 = 'F:/tfrecords/train39_41.tfrecords'
data_path12 = 'F:/tfrecords/train41_43.tfrecords'
data_path13 = 'F:/tfrecords/train43_46.tfrecords'
data_path14 = 'F:/tfrecords/train46_49.tfrecords'
data_path15 = 'F:/tfrecords/train49_53.tfrecords'
data_path16 = 'F:/tfrecords/train53_57.tfrecords'
data_path17 = 'F:/tfrecords/train57_66.tfrecords'
data_path18 = 'F:/tfrecords/train66_100.tfrecords'

with tf.Session() as sess:
    feature = {
        'train/chins': tf.FixedLenFeature([], tf.string),
        'train/pishoonis': tf.FixedLenFeature([], tf.string),
        'train/left_eyes': tf.FixedLenFeature([], tf.string),
        'train/right_eyes': tf.FixedLenFeature([], tf.string),
        'train/noses': tf.FixedLenFeature([], tf.string),
        'train/originals': tf.FixedLenFeature([], tf.string),

        'train/labels': tf.FixedLenFeature([], tf.string)}


    filename_queue = tf.train.string_input_producer([data_path1,data_path2,data_path2,data_path4,data_path5,data_path6,data_path7,data_path8,data_path9,data_path10,
                                                     data_path11,data_path12,data_path13,data_path14,data_path15,data_path16,data_path17,data_path18] , num_epochs=None,shuffle=True,seed=1000000)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features=feature)

    chins = tf.decode_raw(features['train/chins'], tf.uint8)
    pishoonis = tf.decode_raw(features['train/pishoonis'],tf.uint8)
    left_eyes = tf.decode_raw(features['train/left_eyes'], tf.uint8)
    right_eyes = tf.decode_raw(features['train/right_eyes'], tf.uint8)
    noses = tf.decode_raw(features['train/noses'], tf.uint8)
    originals = tf.decode_raw(features['train/originals'], tf.uint8)
    labels = tf.decode_raw(features['train/labels'], tf.float64)


    chins = tf.reshape(chins, [224, 224, 3])

    pishoonis = tf.reshape(pishoonis, [224, 224, 3])
    left_eyes = tf.reshape(left_eyes, [224, 224, 3])
    right_eyes = tf.reshape(right_eyes, [224, 224, 3])
    noses = tf.reshape(noses, [224, 224, 3])
    originals = tf.reshape(originals, [224, 224, 3])
    chins = tf.cast(chins, tf.float32)
    pishoonis = tf.cast(pishoonis,tf.float32)
    left_eyes = tf.cast(left_eyes,tf.float32)
    right_eyes = tf.cast(right_eyes,tf.float32)
    noses = tf.cast(noses,tf.float32)
    originals = tf.cast(originals,tf.float32)
    labels = tf.cast(labels, tf.float32)


    labels = tf.reshape(labels,[1])
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # tf.train.start_queue_runners()

    chin, pishooni, left_eye, right_eye, nose, original , label = tf.train.shuffle_batch(
        [chins, pishoonis, left_eyes, right_eyes, noses, originals, labels],  batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)
    # chin, pishooni, left_eye, right_eye, nose, original, label =tf.train.batch([chins, pishoonis, left_eyes, right_eyes, noses, originals, labels]
    #                                                                            ,batch_size=10,capacity=32,enqueue_many=True)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    counter = 0
    for batch_index in range(500000):
        counter +=1
        print(counter)
        c,p,l,r,n,o,lab = sess.run([chin, pishooni, left_eye, right_eye, nose, original,label])
        print(c.shape)
        print(lab[1])

        # cv2.imshow('salam', np.array(o[1], dtype='uint8'))
        # # cv2.waitKey(100)
        # cv2.imshow('salam1', np.array(n[1], dtype='uint8'))
        # # cv2.waitKey(100)
        # cv2.imshow('salam3', np.array(l[1], dtype='uint8'))
        # # cv2.waitKey(100)
        # cv2.imshow('salam4', np.array(p[1], dtype='uint8'))
        # cv2.waitKey(100)

    coord.request_stop()



    # Wait for threads to stop
    coord.join(threads)
    sess.close()




