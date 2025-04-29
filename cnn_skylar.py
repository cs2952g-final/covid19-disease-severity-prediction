from __future__ import absolute_import
from matplotlib import pyplot as plt

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import random
import math
from visualization import SaliencyMap
from preprocessing import get_data
import sys
import scanpy as sc

#from preprocessing_skylar import cell_data

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
        self.num_classes = 4 # mild, severe, critical, no disease
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main

        # TODO: Initialize all hyperparameters
        # self.input_width = 32
        # self.hidden_layer_size = 100
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

        self.conv1 = tf.keras.layers.Conv2D(32, 5, activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=2)

        self.conv2 = tf.keras.layers.Conv2D(64, 5, activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=2)

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)

        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)

        self.out = tf.keras.layers.Dense(self.num_classes, activation='softmax')


    def batch_normalize(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2], keepdims=False)
        return tf.nn.batch_normalization(inputs, mean, variance, offset=None, scale=None, variance_epsilon=0.00001)

    def call(self, inputs, is_testing):
        """
        Runs a forward pass on an input batch of cell x gene matrices.
        """

        num_classes = inputs.shape[1]
        
        # l1 = tf.keras.layers.Conv1D(32, 5, activation='relu', padding='same')(inputs)
        # l1 = tf.keras.layers.MaxPooling1D(pool_size=2)(l1)
        
        # l2 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(l1)
        # l2 = tf.keras.layers.MaxPooling1D(pool_size=2)(l2)

        # l3 = tf.keras.layers.Conv1D(128, 5, activation='relu', padding='same')(l2)
        # l3 = tf.keras.layers.MaxPooling1D(pool_size=2)(l3)
        
        # l4 = tf.keras.layers.Flatten()(l3)
        # l4 = tf.keras.layers.Dense(256, activation='relu')(l4)
        # l4 = tf.keras.layers.Dropout(0.5)(l4)
        # l4 = tf.keras.layers.Dense(128, activation='relu')(l4)
        # l4 = tf.keras.layers.Dropout(0.5)(l4)

        x = self.conv1(inputs)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x, training= not is_testing)

        x = self.dense2(x)
        x = self.dropout2(x, training= not is_testing)

        return self.out(x)

        #outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(l4)
        
        # # apply conv2d to the cell by gene matrix (x3)
        # l1 = tf.nn.conv2d(inputs, self.filter1, strides = [1, 1, 1, 1], padding="SAME")       
        # l1 = tf.nn.bias_add(l1, self.fbias1)
        # l1 = tf.nn.relu(l1)
        # l1 = tf.nn.max_pool(l1, ksize = 3, strides = [1, 1, 1, 1], padding="SAME")

        # # conv layer 2
        # l2 = tf.nn.conv2d(l1, self.filter2, strides = [1, 1, 1, 1], padding="SAME") #can adjust stride here
        # l2 = tf.nn.bias_add(l2, self.fbias2)
        # l2 = tf.nn.relu(l2)
        # l2 = tf.nn.max_pool(l2, ksize = 3, strides = [1, 1, 1, 1], padding="SAME") #can adjust stride here

        # # conv layer 3
        # l3 = tf.nn.conv2d(l2, self.filter3, strides = [1, 1, 1, 1], padding="SAME") #can adjust stride here
        # l3 = tf.nn.bias_add(l3, self.fbias2)
        # l3 = tf.nn.relu(l3)
        # l3 = tf.nn.max_pool(l3, ksize = 3, strides = [1, 1, 1, 1], padding="SAME") #can adjust stride here
        
        # # conv layer 4
        # l4 = tf.nn.conv2d(l3, self.filter4, strides = [1, 1, 1, 1], padding="SAME")
        # l4 = tf.nn.bias_add(l4, self.fbias4)
        # l4 = tf.nn.relu(l4)

        # #(TBD) reshape the output to make it compatible with dense layers
        # shape_list = tf.Variable(l4).shape
        # # will change if not 3d
        # num_elements = shape_list[1]*shape_list[2]*shape_list[3]
        # l4 = tf.reshape(l4, [shape_list[0], num_elements])

        # #dense layer 1
        # dense1 = tf.matmul(l4,self.w1)+self.b1
        # dense1 = tf.nn.leaky_relu(dense1)
        
        return outputs

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        loss = tf.keras.losses.CategoricalCrossentropy()
        return loss

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
        correct_predictions = tf.equal(tf.argmax(logits), tf.argmax(labels))
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

    loss_fcn = tf.keras.losses.SparseCategoricalCrossentropy()

    for b, b1 in enumerate(range(model.batch_size, train_inputs.shape[0] + 1, model.batch_size)):
        b0 = b1 - model.batch_size

        input_batch = tf.keras.layers.Input(shape=train_inputs.shape, batch_size=model.batch_size)
        label_batch = tf.keras.layers.Input(shape=train_labels.shape, batch_size=model.batch_size)
        #print(train_inputs.shape)
        #batch_dim = train_inputs[b0:b1][0]
        #batch_dim_labels = train_labels[b0:b1][0]
        #train_inputs = tf.expand_dims(train_inputs, axis=0)
        #train_inputs[0] = batch_dim
        #train_inputs = tf.tensor_scatter_nd_update(train_inputs, )

        #input_batch = train_inputs[b0:b1]
        #label_batch = train_labels[b0:b1]

        
        with tf.GradientTape() as tape:
            pred = model(input_batch, is_testing=False)
            label_batch = tf.cast(label_batch, tf.float32)
            loss = loss_fcn(label_batch, pred)

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

def get_severity(annDataObj): 
        '''
        Return all severities of an anndata object as a list.
        '''
        severity_labels = sc.get.obs_df(annDataObj, keys = ["Source"])
        # print(type(severity_labels))

        severity_dict = {
            'HV':0,
            'COVID_MILD':1,
            'COVID_SEV':2, 
            'COVID_CRIT':3, 
        }

        return np.array([severity_dict[severity_label] for severity_label in severity_labels["Source"]]) 

def main():
    '''
    Read in cellxgene matrix!

    Consider printing the loss, training accuracy, and testing accuracy after each epoch
    to ensure the model is training correctly.
    
    :return: None
    '''

    # set up cell-type data 
    #combat_file='/Users/skylarwalters/Desktop/data/COMBAT2022.h5ad'
    #get_data(combat_file, 3, 1000)
    dendrite_training = (sc.read_h5ad('training/dendritic cell_training'))
    dendrite_testing = (sc.read_h5ad('testing/dendritic cell_testing'))
    #dendrite_training, dendrite_testing=get_data(combat_file, 3, 1000)

    dendrite_training_labels = get_severity(dendrite_training)
    dendrite_testing_labels = get_severity(dendrite_testing)

    dendrite_training_dense = dendrite_training.X.todense()
    dendrite_testing_dense = dendrite_testing.X.todense()
    print(dendrite_testing_dense.shape)

    dendrite_final = tf.expand_dims(dendrite_training_dense, axis=-1)
    print(dendrite_final.shape)

    # instantiate model 
    model = Model()

    for epoch in range(1, 3):
        train_acc = train(model, dendrite_final, dendrite_training_labels)
        print(f'Epoch {epoch}, Training Accuracy: {train_acc}')

    test_acc = test(model, dendrite_testing_dense, dendrite_testing_labels)
    print(f'Testing Accuracy: {test_acc}')

    # visualize_loss(model.loss_list)

    # saliency maps 
    # SM = SaliencyMap(model)
    # grads = SM.get_gradients(test_inputs) # check input here?
    # norm_grads = SM.norm_grad(grads)

if __name__ == '__main__':
    main()