import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, DepthwiseConv2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for CIFAR-10 images (32x32 pixels, 3 channels)
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the last dimension
    split_layers = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Define the depthwise separable convolutions with different kernel sizes
    conv1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_layers[0])
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_layers[1])
    conv3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_layers[2])

    # Concatenate the outputs of the three convolutional layers
    concatenated = Concatenate()([conv1, conv2, conv3])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Fully connected layer for classification
    output_layer = Dense(10, activation='softmax')(flattened)  # 10 classes for CIFAR-10

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of using the function to create the model
model = dl_model()
model.summary()