import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Multiply, Concatenate, AveragePooling2D, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Initial feature extraction
    initial_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm = BatchNormalization()(initial_conv)
    relu = Activation('relu')(batch_norm)

    # Compress feature maps using global average pooling
    global_avg_pool = GlobalAveragePooling2D()(relu)

    # Fully connected layers to match dimensions with initial features
    fc1 = Dense(units=64, activation='relu')(global_avg_pool)
    fc2 = Dense(units=64, activation='sigmoid')(fc1)  # Use sigmoid to create gating signals

    # Reshape to match the size of initial feature maps
    reshaped = keras.layers.Reshape(target_shape=(1, 1, 64))(fc2)

    # Weighted feature maps using multiplication
    weighted_features = Multiply()([relu, reshaped])

    # Concatenate weighted features with the input layer
    concatenated = Concatenate()([input_layer, weighted_features])

    # Dimensionality reduction and downsampling
    conv1x1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1x1)

    # Final fully connected layer for classification
    flatten = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)  # CIFAR-10 has 10 classes

    # Model creation
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model