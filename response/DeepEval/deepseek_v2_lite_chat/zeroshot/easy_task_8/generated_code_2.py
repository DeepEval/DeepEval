from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout, Flatten, Dense
from keras.layers import SeparableConv2D


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the datasets to include channels (for Conv2D)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Normalize data to [0, 1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Display a few images from the training dataset
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()


def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))
    
    # Depthwise separable convolutional layer
    x = SeparableConv2D(32, 3, activation='relu')(inputs)
    
    # Depthwise separable convolutional layer
    x = SeparableConv2D(32, 3, activation='relu')(x)
    
    # Max pooling
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Dropout layer to mitigate overfitting
    x = Dropout(0.2)(x)
    
    # 1x1 convolutional layer for feature extraction
    x = SeparableConv2D(64, 1, activation='relu')(x)
    
    # Dropout layer to mitigate overfitting
    x = Dropout(0.2)(x)
    
    # Upsampling to match the original input size
    x = UpSampling2D(size=(2, 2))(x)
    
    # Depthwise separable convolutional layer
    x = SeparableConv2D(32, 3, activation='relu')(x)
    
    # Depthwise separable convolutional layer
    x = SeparableConv2D(32, 3, activation='relu')(x)
    
    # Output layer
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model