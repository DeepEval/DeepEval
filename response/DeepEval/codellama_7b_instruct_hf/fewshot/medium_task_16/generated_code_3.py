import keras
from keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    # Split the input into three groups along the channel dimension
    input_split = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    # Apply 1x1 convolutions to each group independently
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_split[0])
    conv2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_split[1])
    conv3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_split[2])
    # Downsample each group using average pooling
    pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    pool3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
    # Concatenate the downsampled feature maps along the channel dimension
    concat = Concatenate()([pool1, pool2, pool3])
    # Flatten the concatenated feature maps into a one-dimensional vector
    flatten = Flatten()(concat)
    # Pass the flattened feature maps through two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=dense2)
    return model