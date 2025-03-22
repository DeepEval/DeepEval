import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout

def dl_model():
    # Input layer
    input_layer = tf.keras.Input(shape=(28, 28, 1))

    # First convolutional layer with average pooling
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(3, 3), activation='relu')(input_layer)
    x = AveragePooling2D(pool_size=(5, 5))(x)

    # Second convolutional layer (1x1 convolution)
    x = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)

    # Flatten the feature maps
    x = Flatten()(x)

    # First fully connected layer
    x = Dense(128, activation='relu')(x)

    # Dropout layer to mitigate overfitting
    x = Dropout(0.5)(x)

    # Second fully connected layer (output layer)
    output_layer = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()