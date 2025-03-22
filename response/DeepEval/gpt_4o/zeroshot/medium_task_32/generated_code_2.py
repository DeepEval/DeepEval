import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, SeparableConv2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for CIFAR-10 images (32x32 RGB images)
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the last dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Apply depthwise separable convolutions with different kernel sizes to each split
    conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_layer[0])
    conv3 = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(split_layer[1])
    conv5 = SeparableConv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(split_layer[2])

    # Concatenate the results of the convolutional layers
    concatenated = Concatenate()([conv1, conv3, conv5])

    # Flatten the concatenated outputs
    flattened = Flatten()(concatenated)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model