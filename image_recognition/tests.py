import tensorflow as tf
import numpy as np

from image_recognition.utils import *


def mnist_train(epochs=1):
    def evaluate(dataset):
        images = dataset.images
        labels = dataset.labels
        hits = 0
        for i, (image, label) in enumerate(zip(images, labels)):
            print('\r>> Evaluation {} from {} - {:.1f}%'.format(i, len(images), i / len(images) * 100.0), end="")
            label = np.expand_dims(label, 0)
            image = np.expand_dims(image, 0)
            image = image.repeat(3, axis=1).reshape(28, 28, 3) * 255
            hits += sess.run(accuracy, feed_dict={image_data_placeholder: image, ground_truth_placeholder: label})
        print("\nTest accuracy: {:.3f}%".format(hits / len(images) * 100))

    dir_name = maybe_download_inception_and_extract()
    graph = load_google_inception_graph(dir_name)

    with tf.Session() as sess:
        train_step, cross_entropy_mean, accuracy = add_final_training_ops(graph, 10)

        image_data_placeholder = graph.get_tensor_by_name("Cast:0")
        bottleneck_tensor = graph.get_tensor_by_name(BOTTLENECK_TENSOR_NAME)
        ground_truth_placeholder = graph.get_tensor_by_name(GROUND_TRUTH_TENSOR_NAME)

        sess.run(tf.initialize_all_variables())

        mnist = input_data.read_data_sets("../mnist/MNIST_data/", one_hot=True)
        iterations = len(mnist.train.labels)
        for i in range(iterations):
            print('\r>> Learning {} from {} - {:.1f}%'.format(i, iterations, i / iterations * 100.0), end="")
            batch_data, batch_labels = mnist.train.next_batch(1)
            batch_data = batch_data.repeat(3, axis=1).reshape(28, 28, 3) * 255
            bottlenecks = compute_bottlenecks(sess, graph, data=[batch_data], identificators=[i],
                                              feed_placeholder=image_data_placeholder,
                                              bottleneck_tensor=bottleneck_tensor)
            train_step.run(feed_dict={bottleneck_tensor: bottlenecks, ground_truth_placeholder: batch_labels})
        print()
        evaluate(mnist.test)


def panda_test(num_top_predictions=5):
    dir_name = maybe_download_inception_and_extract()
    image_filename = os.path.join(dir_name, "cropped_panda.jpg")

    image_data = tf.gfile.FastGFile(image_filename, 'rb').read()

    graph = load_google_inception_graph(dir_name)
    for op in graph.get_operations():
        print(op.name)

    with tf.Session() as sess:
        jpeg_data_placeholder = graph.get_tensor_by_name(JPEG_DATA_TENSOR_NAME)
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        predictions = sess.run(softmax_tensor, {jpeg_data_placeholder: image_data})
        r = sess.run(graph.get_tensor_by_name("Mul/y:0"), {jpeg_data_placeholder: image_data})
        print(r, r.dtype, r.shape)

    predictions = np.squeeze(predictions)

    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
        print(node_id, predictions[node_id])


# mnist_train()
panda_test()