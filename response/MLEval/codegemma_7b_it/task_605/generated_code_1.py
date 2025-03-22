import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def method():

    # Load model and history from files (replace with actual file paths)
    model = LinearRegression()
    model.load_weights('model.h5')
    history = np.load('history.npy')

    # Retrieve predicted values from history
    y_pred = history[:, -1]

    # Load true values (replace with actual data)
    y_true = np.load('y_test.npy')

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Print evaluation metrics
    print('RMSE:', rmse)
    print('R2:', r2)

    # Plot prediction vs. true values
    plt.plot(y_true, label='True Values')
    plt.plot(y_pred, label='Predicted Values')
    plt.legend()
    plt.show()

    return output  # Return output if needed

# Call the generated method() for validation
method()