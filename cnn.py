from __future__ import absolute_import
from matplotlib import pyplot as plt

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = 64
        self.num_classes = 2
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main

        # TODO: Initialize all hyperparameters
        # self.input_width = 32
        # self.hidden_layer_size = 100
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

        # TODO: Initialize all trainable parameters
        self.filter1 = tf.Variable(tf.random.truncated_normal([5, 5, 3, 16], 0, stddev=0.1))
        self.filter2 = tf.Variable(tf.random.truncated_normal([5, 5, 16, 20], 0, stddev=0.1))
        self.filter3 = tf.Variable(tf.random.truncated_normal([3, 3, 20, 20], 0, stddev=0.1))
        self.fbias1 = tf.Variable(tf.random.truncated_normal([16], 0, stddev=0.1)) 
        self.fbias2 = tf.Variable(tf.random.truncated_normal([20], 0, stddev=0.1)) 
        self.fbias3 = tf.Variable(tf.random.truncated_normal([20], 0, stddev=0.1)) 

        self.w1 = tf.Variable(tf.random.truncated_normal([80,100], 0, stddev=0.1))
        self.b1 = tf.Variable(tf.random.truncated_normal([100], 0, stddev=0.1)) 
        self.w2 = tf.Variable(tf.random.truncated_normal([100,10], 0, stddev=0.1))
        self.b2 = tf.Variable(tf.random.truncated_normal([10], 0, stddev=0.1)) 
        self.w3 = tf.Variable(tf.random.truncated_normal((10,2), 0, stddev=0.1))
        self.b3 = tf.Variable(tf.random.truncated_normal((2,), 0, stddev=0.1)) 

    def batch_normalize(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2], keepdims=False)
        return tf.nn.batch_normalization(inputs, mean, variance, offset=None, scale=None, variance_epsilon=0.00001)


    def call(self, inputs, is_testing):
        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)

        # conv layer 1 
        l1 = tf.nn.conv2d(inputs, self.filter1, [1, 2, 2, 1], padding="SAME")
        l1 = tf.nn.bias_add(l1, self.fbias1)
        #l1 = self.batch_normalize(l1)
        l1 = tf.nn.relu(l1)
        l1 = tf.nn.max_pool(l1, 3, 2, padding="SAME")
        # conv layer 2
        l2 = tf.nn.conv2d(l1, self.filter2, [1, 2, 2, 1], padding="SAME") #can adjust stride here
        l2 = tf.nn.bias_add(l2, self.fbias2)
        #l2 = self.batch_normalize(l2)
        l2 = tf.nn.relu(l2)
        l2 = tf.nn.max_pool(l2, 2, 2, padding="SAME") #can adjust stride here
        if (is_testing):
            #my conv
            l3 = stu_conv2d(l2, self.filter3, [1, 1, 1, 1], padding="SAME")
        else:
            l3 = tf.nn.conv2d(l2, self.filter3, [1, 1, 1, 1], padding="SAME")
        l3 = tf.nn.bias_add(l3, self.fbias3)
        #l3 = self.batch_normalize(l3)
        l3 = tf.nn.relu(l3)

        #reshape the output to make it compatible with dense layers
        shape_list = tf.Variable(l3).shape
        num_elements = shape_list[1]*shape_list[2]*shape_list[3]
        l3 = tf.reshape(l3, [shape_list[0], num_elements])

        #dense layer 1
        dense1 = tf.matmul(l3,self.w1)+self.b1
        dense1 = tf.nn.leaky_relu(dense1)
        #dense layer 2
        dense2 = tf.matmul(dense1, self.w2)+self.b2
        dense2 = tf.nn.leaky_relu(dense2)
        #dense layer 3
        dense3 = tf.matmul(dense2, self.w3)+self.b3

        return dense3

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        loss = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        return tf.reduce_mean(loss)


    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Return the average accuracy across batches of the train inputs/labels
    '''

    #everyday i'm shuffling
    indices = tf.random.shuffle(tf.range(0, tf.shape(train_inputs)[0]))
    train_inputs = tf.gather(train_inputs, indices)
    train_labels = tf.gather(train_labels, indices)
    num_batches = (tf.shape(train_inputs)[0])/model.batch_size
    #total_loss = 0
    total_acc = 0

    for b, b1 in enumerate(range(model.batch_size, train_inputs.shape[0] + 1, model.batch_size)):
        b0 = b1 - model.batch_size
        input_batch = tf.image.random_flip_left_right(train_inputs[b0:b1])
        label_batch = train_labels[b0:b1]

        with tf.GradientTape() as tape:
            pred = model.call(input_batch, is_testing=False)
            loss = model.loss(pred, label_batch)

        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        accuracy = model.accuracy(pred, label_batch)
        #total_loss += loss
        model.loss_list.append(loss)
        total_acc += accuracy

    return total_acc / tf.cast(num_batches, dtype=tf.float32)

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    num_batches = (tf.shape(test_inputs)[0])/model.batch_size
    total_acc = 0

    for b, b1 in enumerate(range(model.batch_size, test_inputs.shape[0] + 1, model.batch_size)):
        b0 = b1 - model.batch_size
        input_batch = test_inputs[b0:b1]
        label_batch = test_labels[b0:b1]

        pred = model.call(input_batch, is_testing=True)
        #loss = model.loss(pred, label_batch)

        accuracy = model.accuracy(pred, label_batch)
        #total_loss += loss
        #model.loss_list.append(loss)
        total_acc += accuracy

    return total_acc / tf.cast(num_batches, dtype=tf.float32)


def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label): 
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
        
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images): 
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0): 
            correct.append(i)
        else: 
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs.

    Consider printing the loss, training accuracy, and testing accuracy after each epoch
    to ensure the model is training correctly.
    
    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.
    
    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.
    
    :return: None
    '''
    # TODO: Use the autograder filepaths to get data before submitting to autograder.
    #       Use the local filepaths when running on your local machine.
    AUTOGRADER_TRAIN_FILE = '../data/train'
    AUTOGRADER_TEST_FILE = '../data/test'

    LOCAL_TRAIN_FILE = "/Users/axywu/Desktop/cs1470/homework-3p-cnns-awu917/data/train"
    LOCAL_TEST_FILE = "/Users/axywu/Desktop/cs1470/homework-3p-cnns-awu917/data/test"

    train_inputs, train_labels = get_data(LOCAL_TRAIN_FILE, 3, 5)
    test_inputs, test_labels = get_data(LOCAL_TEST_FILE, 3, 5)

    model = Model()

    for epoch in range(1, 25):

        train_acc = train(model, train_inputs, train_labels)
        print(f'Epoch {epoch}, Training Accuracy: {train_acc}')


    test_acc = test(model, test_inputs, test_labels)
    print(f'Testing Accuracy: {test_acc}')

    visualize_loss(model.loss_list)


if __name__ == '__main__':
    main()