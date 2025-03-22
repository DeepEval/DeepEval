import keras
from keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split input channels
    x = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    # 1x1 convolutions on each channel group
    x = [Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(group) for group in x]

    # Average pooling
    x = [AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(group) for group in x]

    # Concatenate feature maps
    x = Concatenate()(x)

    # Flatten and fully connected layers
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model