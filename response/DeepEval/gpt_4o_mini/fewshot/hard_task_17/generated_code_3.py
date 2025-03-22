import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Conv2D, MaxPooling2D, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Block 1
    avg_pool = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=32, activation='relu')(avg_pool)
    dense2 = Dense(units=32, activation='sigmoid')(dense1)
    reshaped_weights = Reshape((1, 1, 32))(dense2)  # Reshape to match input dimensions
    weighted_output = Multiply()([input_layer, reshaped_weights])  # Multiply input with the weights

    # Block 2
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    max_pool = MaxPooling2D(pool_size=(2, 2))(conv2)  # Apply max pooling after convolutional layers

    # Combine outputs from Block 1 and Block 2
    combined_output = Add()([weighted_output, max_pool])  # Element-wise addition of outputs from Block 1 and Block 2

    # Classification layers
    flatten = Flatten()(combined_output)
    dense3 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense3)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model