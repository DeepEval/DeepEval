import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def block(input_tensor):
    # Elevate dimension
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    # Extract features
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depth_activation=True)(path1)
    # Reduce dimension
    path3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
    # Add block's input
    output_tensor = Concatenate()([path1, path2, path3, input_tensor])
    return output_tensor


def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    # Initial convolutional layer
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    # Max pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    # Three branches
    branch1 = block(input_tensor=max_pooling)
    branch2 = block(input_tensor=branch1)
    branch3 = block(input_tensor=branch2)

    # Concatenate outputs
    concat = Concatenate()([branch1, branch2, branch3])

    # Batch normalization
    bath_norm = BatchNormalization()(concat)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model