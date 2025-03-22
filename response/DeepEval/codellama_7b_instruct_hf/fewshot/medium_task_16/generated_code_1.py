import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split input into three groups along the channel dimension
    groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    # Apply 1x1 convolutions to each group independently
    convs = [Conv2D(filters=int(32 / 3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group) for group in groups]

    # Apply downsampling via an average pooling layer with consistent parameters
    pools = [MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv) for conv in convs]

    # Concatenate the three groups of feature maps along the channel dimension
    concat = Concatenate()(pools)

    # Flatten the concatenated feature maps into a one-dimensional vector
    flat = Flatten()(concat)

    # Pass the flattened feature maps through two fully connected layers for classification
    output = Dense(units=10, activation='softmax')(flat)

    model = keras.Model(inputs=input_layer, outputs=output)
    return model