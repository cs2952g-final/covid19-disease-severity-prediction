import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np 
import scanpy as sc
import anndata as ad
from sklearn.preprocessing import normalize

filter_size=3
pool_size=2
epochs=10
batch_size=32
cell_type='CD4-positive, alpha-beta T cell'

def cnn_model(input_shape, num_classes):
    """
    Build a CNN model for cell x gene matrix input.
    """
    inputs = layers.Input(shape=input_shape)

    # Convolutional Block 1
    x = layers.Conv1D(32, filter_size, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=pool_size)(x)

    # Convolutional Block 2
    x = layers.Conv1D(64, filter_size, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=pool_size)(x)

    # Convolutional Block 3
    x = layers.Conv1D(128, filter_size, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=pool_size)(x)

    # Flatten and Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Output Layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # or 'categorical_crossentropy' if labels are one-hot
                  metrics=['accuracy'])
    return model

def get_severity(annDataObj): 
        '''
        Return all severities of an anndata object as a list.
        '''
        severity_labels = sc.get.obs_df(annDataObj, keys = ["Source"])

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
    b_cell_training = (sc.read_h5ad(f'training/{cell_type}_training'))
    b_cell_testing = (sc.read_h5ad(f'testing/{cell_type}_testing'))

    b_cell_training_labels = get_severity(b_cell_training)
    b_cell_testing_labels = get_severity(b_cell_testing)


    #b_cell_training_dense = normalize(b_cell_training.X, norm='l1', axis=1).todense()
    #b_cell_testing_dense = normalize(b_cell_testing.X, norm='l1', axis=1).todense()

    b_cell_training_dense = b_cell_training.X.todense()
    b_cell_testing_dense = b_cell_testing.X.todense()

    b_cell_final_training = tf.expand_dims(b_cell_training_dense, axis=-1)
    #b_cell_final_training_labels = tf.expand_dims(b_cell_training_labels, axis=-1)

    b_cell_final_testing = tf.expand_dims(b_cell_testing_dense, axis=-1)
    #b_cell_final_testing_labels = tf.expand_dims(b_cell_testing_labels, axis=-1)

    # instantiate model 
    num_classes = 4  # different severity levels (HV, mild, severe, critical)

    input_shape = b_cell_final_training.shape[1:]
    print(input_shape)
    print(b_cell_training_labels.shape)
    
    model = cnn_model(input_shape, num_classes)
    model.summary()

    # # train model on cell type
    model.fit(b_cell_final_training, b_cell_training_labels, epochs=epochs, batch_size=batch_size)

    # test model on trained cell type
    test_loss, test_acc = model.evaluate(b_cell_final_testing, b_cell_testing_labels)
    print("Test Accuracy:", test_acc)

    # saliency maps 
    # SM = SaliencyMap(model)
    # grads = SM.get_gradients(test_inputs) # check input here?
    # norm_grads = SM.norm_grad(grads)



