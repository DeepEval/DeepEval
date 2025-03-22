import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial 1x1 Convolutional Layer
    initial_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # First Branch: Local Feature Extraction with Two 3x3 Convolutional Layers
    branch1_conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)
    branch1_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1_conv1)

    # Second Branch: Average Pooling followed by 3x3 Convolution and Transposed Convolution
    branch2_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(initial_conv)
    branch2_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_pool)
    branch2_upsample = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch2_conv)

    # Third Branch: Average Pooling followed by 3x3 Convolution and Transposed Convolution
    branch3_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(initial_conv)
    branch3_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3_pool)
    branch3_upsample = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch3_conv)

    # Concatenate Outputs from All Branches
    concatenated = Concatenate()([branch1_conv2, branch2_upsample, branch3_upsample])

    # 1x1 Convolutional Layer for Refinement
    refined_output = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Fully Connected Layer for Classification
    flatten_layer = Flatten()(refined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model