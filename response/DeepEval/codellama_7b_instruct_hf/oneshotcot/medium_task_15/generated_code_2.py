import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Concatenate, BatchNormalization, ReLU

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Batch normalization and ReLU activation
    batch_norm = BatchNormalization()(conv)
    relu = ReLU()(batch_norm)

    # Global average pooling
    global_avg_pool = GlobalAveragePooling2D()(relu)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Reshape output to match the size of the initial features
    reshaped_output = Flatten()(dense2)

    # Multiply the output with the initial features to generate weighted feature maps
    weighted_feature_maps = reshaped_output * input_layer

    # Concatenate the weighted feature maps with the input layer
    concatenated_features = Concatenate()([weighted_feature_maps, input_layer])

    # 1x1 convolution and average pooling
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated_features)
    avg_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1x1)

    # Fully connected layer
    dense = Dense(units=10, activation='softmax')(avg_pool)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model