import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Add

def dl_model():
    # Input layer for CIFAR-10 images (32x32 RGB images)
    input_layer = Input(shape=(32, 32, 3))

    # Pathway 1: Two blocks of convolution followed by average pooling
    path1_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path1_conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1_conv1)
    path1_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(path1_conv2)

    # Pathway 2: A single convolutional layer
    path2_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combining both pathways using addition
    combined = Add()([path1_pool, path2_conv])

    # Flatten the combined output
    flattened = Flatten()(combined)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model