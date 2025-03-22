import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    groups = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    # Apply 1x1 convolutions to each group independently
    conv1 = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='same')(groups[0])
    conv2 = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='same')(groups[1])
    conv3 = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='same')(groups[2])

    # Downsample each group via an average pooling layer
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)

    # Concatenate the three downsampled feature maps along the channel dimension
    concatenated = Lambda(lambda x: tf.concat(x, axis=-1))([pool1, pool2, pool3])

    # Flatten the concatenated feature maps into a one-dimensional vector
    flattened = Flatten()(concatenated)

    # Pass the flattened feature maps through two fully connected layers for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model