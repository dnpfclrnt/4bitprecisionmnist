import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

LOGDIR = './mnist_tutorial/'


def conv_layer(input, conv_kernel, conv_strides, conv_padding="SAME", name="Conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal(conv_kernel, mean=0.0, stddev=0.1), name="W")

        b = tf.Variable(tf.constant(0.0, shape=[conv_kernel[3]]), name='b')

        conv = tf.nn.conv2d(input, w, strides=conv_strides, padding=conv_padding)

        batch_norm = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001)(conv)

        act = tf.nn.leaky_relu(batch_norm+b)

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)

        return act


def pool_layer(input, pool_kernel, pool_strides, pool_padding="VALID", name="Pool"):
    with tf.name_scope(name):
        pool = tf.nn.max_pool(input, ksize=pool_kernel, strides=pool_strides, padding=pool_padding)
        return pool


def fc_layer(input, size_in, size_out, dropoutProb=None, name="FC"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.0, shape=[size_out]), name="B")

        batch_norm = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001)(input)

        act=tf.nn.leaky_relu(tf.matmul(batch_norm, w) + b)

        if dropoutProb is not None:
            act = tf.nn.dropout(act, dropoutProb, name="dropout")

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


def CNN_model(x, n_classes, nShape, nchannels, dropout):
    x = tf.reshape(x, [-1, nShape, nShape, nchannels])
    tf.summary.image('input', x, 10)

    conv1 = conv_layer(x, [3, 3, nchannels, 32], [1, 1, 1, 1], name="Conv_1")
    pool1 = pool_layer(conv1, [1, 2, 2, 1], [1, 2, 2, 1], name="Pool_1")

    conv2 = conv_layer(pool1, [3, 3, 32, 64], [1, 1, 1, 1], name="Conv_2")
    pool2 = pool_layer(conv2, [1, 2, 2, 1], [1, 2, 2, 1], name="Pool_2")

    conv3 = conv_layer(pool2, [3, 3, 64, 128], [1, 1, 1, 1], name="Conv_3")
    pool3 = pool_layer(conv3, [1, 2, 2, 1], [1, 2, 2, 1], name="Pool_3")

    pool3_shape = pool3.get_shape().as_list()
    flattened = tf.reshape(pool3, [-1, pool3_shape[1]*pool3_shape[2]*pool3_shape[3]])

    flattened_shape = flattened.get_shape().as_list()
    fc1 = fc_layer(flattened, flattened_shape[1], 1024, dropoutProb=dropout, name="FC_1")

    fc2 = fc_layer(fc1, 1024, 512, dropoutProb=dropout, name="FC_2")

    Prediction = fc_layer(fc2, 512, n_classes, name="Predicted_Digit")
    return Prediction


def Trainer():

    minibatch_size = 128
    lear_rate = 0.0001
    bta1 = 0.9
    bta2 = 0.999
    epsln = 0.00000001

    keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
    ph_x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    ph_y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")


    Predicted_Digit_logits=CNN_model(ph_x, n_classes=10, nShape=28, nchannels=1, dropout=keep_prob)

    with tf.name_scope("Loss"):
        Digit_Softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Predicted_Digit_logits, labels=ph_y, name="Digit_softmax_loss"))

        tf.summary.scalar("Digit_Softmax_loss", Digit_Softmax_loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=lear_rate, beta1=bta1, beta2=bta2, epsilon=epsln).minimize(Digit_Softmax_loss)

    with tf.name_scope("Digit_classification_Accuracy"):
        output_digit_softmax = tf.nn.softmax(Predicted_Digit_logits, name="output_digit_softmax")
        Predicted_Digit = tf.cast(tf.argmax(output_digit_softmax, 1), tf.int32)
        Label = tf.cast(tf.argmax(ph_y, 1), tf.int32)
        Correct_Class = tf.equal(Predicted_Digit, Label)
        Classification_accuracy = tf.reduce_mean(tf.cast(Correct_Class, tf.float32))
        tf.summary.scalar("Digit_Classification_Acc", Classification_accuracy)
    summ = tf.summary.merge_all()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        Train_writer = tf.summary.FileWriter(LOGDIR + '/Train', sess.graph)
        Test_writer = tf.summary.FileWriter(LOGDIR + '/Test')

        sess.run(tf.global_variables_initializer())
        for itr in range(2001):
            batch = mnist.train.next_batch(batch_size=minibatch_size)
            _ = sess.run([optimizer], feed_dict={ph_x: batch[0], ph_y: batch[1], keep_prob: 0.5})
            if itr %10 == 0:
                [train_acc, s] = sess.run([Classification_accuracy, summ], feed_dict={ph_x: batch[0], ph_y:batch[1], keep_prob:1.0})
                Train_writer.add_summary(s, itr)
                [test_acc, s] = sess.run([Classification_accuracy, summ], feed_dict={ph_x:mnist.test.images, ph_y:mnist.test.labels, keep_prob:1.0})

                print('@ iteration: %i, Training ACC. = %.6f, Test Acc. = %.6f'%(itr, train_acc, test_acc))
                Test_writer.add_summary(s, itr)
                saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), itr)
