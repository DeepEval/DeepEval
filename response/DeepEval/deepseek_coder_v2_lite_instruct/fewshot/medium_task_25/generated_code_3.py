import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define Path 1: Single 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define Path 2: Average pooling followed by a 1x1 convolution
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pool)

    # Define Path 3: 1x1 convolution followed by two parallel 1x3 and 3x1 convolutions
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1x3 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)
    conv3x1 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv1x1)
    path3 = Concatenate()([conv1x3, conv3x1])

    # Define Path 4: 1x1 convolution followed by a 3x3 convolution, then followed by two parallel 1x3 and 3x1 convolutions
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1x3_3x1 = Concatenate()([Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv3x3),
                                Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv3x3)])
    path4 = Concatenate()([conv1x3_3x1, conv3x3])

    # Concatenate the outputs of these paths
    multi_scale_feature = Concatenate()([path1, path2, path3, path4])

    # Flatten the result
    flatten_layer = Flatten()(multi_scale_feature)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model