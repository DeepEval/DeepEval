import numpy as np
from sklearn.metrics import confusion_matrix

def method(y_true, y_pred):
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate the pixel-level IoU
    iou_pixel = np.diag(cm) / (np.sum(cm, axis=0) + np.sum(cm, axis=1) - np.diag(cm))

    # Calculate the mean IoU
    iou_mean = np.mean(iou_pixel)

    # Calculate the IoU score
    iou_score = iou_mean * (1 - iou_mean)

    # Print the results
    print("Pixel-level IoU:", iou_pixel)
    print("Mean IoU:", iou_mean)
    print("IoU score:", iou_score)

    # Return the output
    return iou_score