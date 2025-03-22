import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer with shape (32, 32, 64)
    input_shape = (32, 32, 64)
    inputs = Input(shape=input_shape)

    # 1x1 Convolutional layer to compress input channels
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)

    # Parallel convolutional layers
    # 1x1 Convolution
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
    
    # 3x3 Convolution
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)

    # Concatenate the results of the two parallel layers
    x = Concatenate()([conv1, conv2])

    # Flatten the feature map into a one-dimensional vector
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)  # Assuming 10 classes for classification

    # Create the model
    model = Model(inputs=inputs, outputs=x)

    return model

# Example usage
model = dl_model()
model.summary()