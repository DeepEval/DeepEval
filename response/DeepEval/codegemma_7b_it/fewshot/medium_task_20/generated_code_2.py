import keras
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: 1x1 convolution
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 2: Two 3x3 convolutions after 1x1 convolution
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
    conv2_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_2)

    # Path 3: Single 3x3 convolution after 1x1 convolution
    conv3_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_1)

    # Path 4: Max pooling followed by 1x1 convolution
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    conv4_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(max_pool)

    # Concatenate outputs from all paths
    concat_layer = concatenate([conv1_1, conv2_3, conv3_2, conv4_1])

    # Flatten and pass through dense layer
    flatten_layer = Flatten()(concat_layer)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Create Keras model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])