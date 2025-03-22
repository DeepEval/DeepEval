import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from sklearn.metrics import roc_auc_score as auc_score

# Assume the following custom function is defined elsewhere in the code
def my_iou_metric_2(preds, valid):
    preds_binary = (preds > 0.5).astype(int)
    label_binary = valid
    return np.sum((preds_binary == label_binary).astype(int)) / (np.sum(preds_binary + label_binary))

def method():
    # Assuming X is the input data and y is the target variable
    X = np.random.rand(100, 10)  # Sample input data
    y = np.random.randint(0, 2, 100)  # Sample target variable

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model and train it
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(X_val)

    # Calculate the default threshold for pixel prediction (0 instead of 0.5)
    threshold = 0

    # Make predictions using the threshold
    y_pred_threshold = (y_pred > threshold).astype(int)

    # Calculate the IoU metric using the custom function
    iou_metric = my_iou_metric_2(y_pred_threshold, y_val)

    # Calculate other metrics (F1 score, PSNR, MSE, AUC score)
    f1 = f1_score(y_val, y_pred)
    psnr_value = psnr(y_val, y_pred)
    mse_value = mse(y_val, y_pred)
    auc = auc_score(y_val, y_pred)

    # Return the output as a dictionary
    output = {
        'iou_metric': iou_metric,
        'f1_score': f1,
        'psnr_value': psnr_value,
       'mse_value': mse_value,
        'auc': auc
    }

    return output

# Call the generated'method()' for validation
output = method()
print(output)