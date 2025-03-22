import keras
from keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    
    input_tensor = Input(shape=(32, 32, 3))

    # Split the input channels
    split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=2))(input_tensor)

    # Apply 1x1 convolutions to each group
    group1 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
    group2 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
    group3 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])

    # Downsample each group with average pooling
    pooled1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(group1)
    pooled2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(group2)
    pooled3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(group3)

    # Concatenate the pooled groups
    combined = Concatenate(axis=2)([pooled1, pooled2, pooled3])

    # Flatten and pass through dense layers
    flatten = Flatten()(combined)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_tensor, outputs=output_layer)

    return model