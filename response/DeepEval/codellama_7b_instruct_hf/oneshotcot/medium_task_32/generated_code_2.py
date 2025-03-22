import keras
from keras.layers import Input, Lambda, Conv2D, DepthwiseSeparableConv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split input into three groups along the last dimension
    input_split = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    # Feature extraction layers
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_split[0])
    conv1_3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_split[1])
    conv1_5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same')(input_split[2])

    # Depthwise separable convolutional layers
    dws_conv1_1 = DepthwiseSeparableConv2D(filters=32, kernel_size=(1, 1), padding='same')(conv1_1)
    dws_conv1_3 = DepthwiseSeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')(conv1_3)
    dws_conv1_5 = DepthwiseSeparableConv2D(filters=32, kernel_size=(5, 5), padding='same')(conv1_5)

    # Max pooling layers
    max_pool1_1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(dws_conv1_1)
    max_pool1_3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(dws_conv1_3)
    max_pool1_5 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(dws_conv1_5)

    # Concatenate feature maps
    concatenated_feature_maps = Concatenate()([max_pool1_1, max_pool1_3, max_pool1_5])

    # Flatten and fully connected layers
    flattened_features = Flatten()(concatenated_feature_maps)
    dense1 = Dense(units=128, activation='relu')(flattened_features)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create and return model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model