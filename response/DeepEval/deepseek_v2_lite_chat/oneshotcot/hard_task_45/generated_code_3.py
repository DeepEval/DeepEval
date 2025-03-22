import keras
from keras.layers import Lambda, Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    # Split the input into 3 groups along the last dimension
    split1 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    split1_1 = Lambda(lambda x: keras.backend.reshape(x, (-1, 3, 11, 11)))(split1)

    # Apply depthwise separable convolutional layers with 1x1, 3x3, and 5x5 kernels
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split1_1)
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Second Block
    split2 = Lambda(lambda x: keras.backend.split(x, 2, axis=-1))(input_layer)
    split2_1 = Lambda(lambda x: keras.backend.reshape(x, (-1, 2, 11, 11)))(split2)

    # Apply different configurations of branches
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split2_1[0])
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split2_1[1])
    pool2 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)

    # Concatenate the outputs from the first block
    concat1 = Concatenate(axis=-1)([conv1, conv2_1, conv2_2])

    # Flatten and fully connected layers
    flatten = Flatten()(concat1)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model