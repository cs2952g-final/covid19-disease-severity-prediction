import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scanpy as sc
import numpy as np
from new_cnn import get_severity

def main():
    '''
    Read in cellxgene matrix!

    Consider printing the loss, training accuracy, and testing accuracy after each epoch
    to ensure the model is training correctly.
    
    :return: None
    '''
    # read in cell type data

    init_training = (sc.read_h5ad('training/hematopoietic stem cell_training'))
    init_testing = (sc.read_h5ad('testing/hematopoietic stem cell_testing'))

    training_labels = get_severity(init_training)
    testing_labels = get_severity(init_testing)

    training_dense = init_training.X.todense()
    testing_dense = init_testing.X.todense()
    
    # shape check:
    print(training_dense.shape)

    # create basic mlp
    model = keras.Sequential([
        layers.Input(shape=(training_dense.shape[1],)),
        layers.Dense(256, activation='relu'),  
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),     
        layers.Dropout(0.5),   
        layers.Dense(4, activation='softmax')  
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    model.fit(training_dense, training_labels, epochs=3, batch_size=32)
    
    loss, accuracy = model.evaluate(testing_dense, testing_labels)
    print(f'Testing Accuracy: {accuracy}')
    print(f'Testing Loss: {loss}')

if __name__ == '__main__':
    main()