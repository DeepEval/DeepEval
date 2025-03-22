import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first block
    x = Conv2D(32, kernel_size=(1, 1), activation='relu')(input_layer)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(32, kernel_size=(5, 5), activation='relu')(x)
    x = Concatenate()([x, x, x])

    # Define the second block
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x[0])
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x[1])
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x[2])
    x = Concatenate()([x, x, x])

    # Define the global average pooling layer
    x = GlobalAveragePooling2D()(x)

    # Define the fully connected layer
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Return the constructed model
    return keras.Model(input_layer, x)