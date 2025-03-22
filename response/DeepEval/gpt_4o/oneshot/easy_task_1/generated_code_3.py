import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Input layer with shape matching the MNIST dataset images (28x28x1)
    input_layer = Input(shape=(28, 28, 1))
    
    # First convolutional layer followed by a max pooling layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Second convolutional layer followed by a max pooling layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(max_pool1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Additional convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(max_pool2)
    
    # Flattening the feature maps into a one-dimensional vector
    flatten_layer = Flatten()(conv3)
    
    # Two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Creating the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model