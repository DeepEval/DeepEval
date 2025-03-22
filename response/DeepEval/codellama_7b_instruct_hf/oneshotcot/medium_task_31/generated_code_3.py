import keras
from keras.layers import Input, Lambda, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    # Split the input image into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    # Apply different convolutional kernels to each group
    conv1_layer = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv3_layer = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    conv5_layer = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layer[2])
    # Concatenate the outputs from the three groups
    concat_layer = Concatenate()([conv1_layer, conv3_layer, conv5_layer])
    # Flatten the concatenated output
    flatten_layer = Flatten()(concat_layer)
    # Add two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    # Return the constructed model
    model = keras.Model(inputs=input_layer, outputs=dense2)
    return model