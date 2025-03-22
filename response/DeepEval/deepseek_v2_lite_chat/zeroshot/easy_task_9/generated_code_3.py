import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense
from keras.layers import Concatenate
from keras.optimizers import Adam

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Model architecture
def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))
    
    # 1x1 convolutional layer to increase dimensionality
    x = Conv2D(64, (1, 1), activation='relu')(inputs)
    
    # Depthwise separable convolutional layer
    x = DepthwiseConv2D(kernel_size=(3, 3), activation='relu')(x)
    
    # 1x1 convolutional layer for dimensionality reduction
    x = Conv2D(32, (1, 1), activation='relu')(x)
    
    # Add the reduced-dimensionality features to the original input
    x = Add()([x, inputs])
    
    # Flatten layer
    x = Flatten()(x)
    
    # Fully connected layer for classification
    outputs = Dense(10, activation='softmax')(x)
    
    # Model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    
    return model

# Build the model
model = dl_model()

# Display model summary
model.summary()