import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Batch normalization and ReLU activation
    batch_norm = BatchNormalization()(conv)
    relu = Activation('relu')(batch_norm)

    # Global average pooling
    pool = GlobalAveragePooling2D()(relu)

    # Two fully connected layers
    dense1 = Dense(units=128, activation='relu')(pool)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Reshape the output
    reshape = Reshape((10, 1))(dense2)

    # Multiply the output with the input
    multiply = Multiply()([input_layer, reshape])

    # Concatenate the output with the input
    concatenate = Concatenate()([input_layer, multiply])

    # 1x1 convolution and average pooling
    conv1 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenate)
    pool1 = AveragePooling2D(pool_size=(2, 2), strides=2)(conv1)

    # Flatten the output
    flatten = Flatten()(pool1)

    # Fully connected layer
    dense3 = Dense(units=64, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model