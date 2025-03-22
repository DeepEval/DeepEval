import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split input into three groups along the channel dimension
    groups = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)

    # Apply 1x1 convolution to each group independently
    groups = [Conv2D(filters=int(x.shape[-1]/3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(g) for g in groups]

    # Downsample each group using average pooling
    groups = [MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(g) for g in groups]

    # Concatenate the downsampled groups along the channel dimension
    output_tensor = Concatenate()(groups)

    # Flatten the output and pass through two fully connected layers for classification
    flattened_output = Flatten()(output_tensor)
    dense1 = Dense(units=128, activation='relu')(flattened_output)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model