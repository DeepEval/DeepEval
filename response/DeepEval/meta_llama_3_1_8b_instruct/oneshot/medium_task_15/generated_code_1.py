import keras
from keras.layers import Input, Conv2D, BatchNormalization, GlobalAveragePooling2D, Reshape, Multiply, Concatenate, AveragePooling2D, Dense

def dl_model():
    """
    Creates a deep learning model for image classification using the CIFAR-10 dataset.
    """

    # Input layer with shape (32, 32, 3) for CIFAR-10 dataset
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layer to extract initial features
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    conv = BatchNormalization()(conv)
    conv = keras.layers.Activation('relu')(conv)

    # Compress feature maps using global average pooling
    global_avg_pool = GlobalAveragePooling2D()(conv)

    # Adjust dimensions to match the channels of the initial features
    reshape_layer = Reshape((32, 3))(global_avg_pool)

    # Multiply weighted feature maps with the initial features
    multiply_layer = Multiply()([input_layer, reshape_layer])

    # Concatenate weighted feature maps with the input layer
    concat_layer = Concatenate()([input_layer, multiply_layer])

    # Reduce dimensionality and downsample the feature
    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(concat_layer)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1x1)

    # Single fully connected layer to output the classification results
    output_layer = Dense(units=10, activation='softmax')(avg_pool)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model