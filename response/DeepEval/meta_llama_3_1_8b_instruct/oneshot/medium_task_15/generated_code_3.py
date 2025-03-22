import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 dataset has 32x32 images with 3 color channels

    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm = BatchNormalization()(conv)

    # Global average pooling to compress the feature maps
    avg_pooling = AveragePooling2D(pool_size=(8, 8))(batch_norm)

    # Flatten the output
    flatten_layer = Flatten()(avg_pooling)

    # Two fully connected layers with ReLU activation
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Reshape the output to match the size of the initial feature maps
    reshape_layer = Reshape((8, 8, 64))(dense2)

    # Concatenate the reshaped output with the initial feature maps
    concat_layer = Concatenate()([batch_norm, reshape_layer])

    # 1x1 convolution to reduce dimensionality
    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_layer)

    # Average pooling to downsample the feature maps
    avg_pooling2 = AveragePooling2D(pool_size=(8, 8))(conv1x1)

    # Flatten the output
    flatten_layer2 = Flatten()(avg_pooling2)

    # Output layer with softmax activation
    output_layer = Dense(units=10, activation='softmax')(flatten_layer2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model