"""
file to create saliency map visualization using the backpropagated gradients from our CNNs.
references: https://github.com/arazd/saliency-tensorflow2/blob/master/methods/base.py
"""

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

class SaliencyMap():
    def __init__(self, model):
        """Constructs a Vanilla Gradient Map by computing dy/dx.

        Args:
            model: The TensorFlow model used to evaluate Gradient Map.
                model takes image as input and outputs probabilities vector.
        """
        self.model = model


    def get_top_predicted_idx(self, image):
        """Outputs top predicted class for the input image.

        Args:
            img_processed: numpy image array in NHWC format, pre-processed according
                to the defined model standard.

        Returns:
            Index of the top predicted class for the input image.
        """
        preds = self.model.predict(image)
        top_pred_idx = tf.argmax(preds[0])
        return top_pred_idx


    def get_gradients(self, image):
        """Computes the gradients of outputs w.r.t input image.

        Args:
            image: numpy image array in NHWC format, pre-processed according
                to the defined model standard.

        Returns:
            Gradients of the predictions w.r.t image (same shape as input image)
        """
        image = tf.convert_to_tensor(image)
        top_pred_idx = self.get_top_predicted_idx(image)

        with tf.GradientTape() as tape:
            tape.watch(image)
            preds = self.model(image)
            top_class = preds[:, top_pred_idx]

        grads = tape.gradient(top_class, image)
        return grads


    def norm_grad(self, grad_x):
        """Normalizes gradient to the range between 0 and 1
        (for visualization purposes).

        Args:
            grad_x: numpy gradients array.

        Returns:
            Gradients of the predictions w.r.t image (same shape as input image)
        """
        abs_grads = np.abs(grad_x)
        grad_max_ = np.max(abs_grads, axis=2)[0]
        arr_min, arr_max  = np.min(grad_max_), np.max(grad_max_)
        normalized_grad = (grad_max_ - arr_min) / (arr_max - arr_min + 1e-18)
        normalized_grad = normalized_grad.reshape(1,grad_x.shape[1],grad_x.shape[2],1)

        return normalized_grad


    def get_top_genes(self, norm_grads, gene_names, num_genes): 
        
        # fix mess of list type
        new_norm_grads = []
        for n in range(len(norm_grads[0])): 
            new_grad = float(str(norm_grads[0][n]).replace("[", "").replace("]", ""))
            new_norm_grads.append(new_grad)

        # get top n gene indices based on highest gradient values (and their indices)
        top_idx = np.argsort(new_norm_grads)[-num_genes:]
        top_values = [new_norm_grads[i] for i in top_idx]
       
        # use those indices to get gene names form gene name list
        top_genes = [gene_names[idx] for idx in top_idx]

        return top_genes
        