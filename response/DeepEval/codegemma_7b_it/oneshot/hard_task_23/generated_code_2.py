import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer for CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Local feature extraction branch
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)

    # Downsampling and upsampling branches
    avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    conv3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool1)
    conv3_2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv3_1)

    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    conv4_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool2)
    conv4_2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv4_1)

    # Concatenation and refinement
    concat = Concatenate()([conv2_2, conv3_2, conv4_2])
    conv5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)

    # Flatten and fully connected layer
    flatten_layer = Flatten()(conv5)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model