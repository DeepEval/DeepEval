import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    # Split the input into three groups along the channel dimension
    input_groups = Lambda(tf.split, [input_layer, 3], axis=-1)
    # Apply 1x1 convolutions to each group independently
    group1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[0])
    group2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[1])
    group3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[2])
    # Downsample each group using average pooling
    group1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(group1)
    group2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(group2)
    group3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(group3)
    # Concatenate the three groups along the channel dimension
    concatenated = Concatenate()([group1, group2, group3])
    # Flatten the concatenated feature maps into a one-dimensional vector
    flattened = Flatten()(concatenated)
    # Pass the flattened feature maps through two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model