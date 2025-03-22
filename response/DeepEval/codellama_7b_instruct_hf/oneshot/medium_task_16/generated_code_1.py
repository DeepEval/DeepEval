import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    x1, x2, x3 = tf.split(input_layer, 3, axis=3)

    # Apply 1x1 convolutions to each group independently
    x1 = Conv2D(filters=int(input_layer.shape[3] / 3), kernel_size=1, strides=1, padding='same')(x1)
    x2 = Conv2D(filters=int(input_layer.shape[3] / 3), kernel_size=1, strides=1, padding='same')(x2)
    x3 = Conv2D(filters=int(input_layer.shape[3] / 3), kernel_size=1, strides=1, padding='same')(x3)

    # Apply downsampling via average pooling layers with consistent parameters
    x1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x1)
    x2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x2)
    x3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x3)

    # Concatenate the resulting feature maps along the channel dimension
    x = Concatenate()([x1, x2, x3])

    # Flatten the concatenated feature maps into a one-dimensional vector
    x = Flatten()(x)

    # Pass the flattened feature maps through two fully connected layers for classification
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=x)

    return model