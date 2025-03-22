import numpy as np
import tensorflow as tf

def my_iou_metric_2(y_true, y_pred):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    union = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true + y_pred, 0, 1)))
    iou = intersection / union
    return iou

def method():
    # Define the ground truth and predicted labels
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.3, 0.7, 0.2, 0.8])

    # Set the default threshold to 0.5
    threshold = 0.5

    # Calculate the IOU using the default threshold
    iou_default = my_iou_metric_2(y_true, y_pred >= threshold)

    # Calculate the IOU using the custom threshold (0)
    iou_custom = my_iou_metric_2(y_true, y_pred)

    # Print the results
    print("Default threshold IOU:", iou_default)
    print("Custom threshold (0) IOU:", iou_custom)

# Call the method for validation
method()