import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPool2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three channel groups
    channel_splits = tf.split(input_layer, num_or_size_splits=3, axis=-1)
    split_layers = [Lambda(lambda x: x[:, :, i, :, :])(channel_splits[i]) for i in range(3)]

    # Define separable convolutional layers
    def separable_conv(filters, size, name, rate=0):
        conv = Conv2D(filters, (size, size), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(split_layers[0])
        conv = Conv2D(filters, (1, 1), padding='same', activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv)
        return conv

    conv1 = separable_conv(32, 1, 'conv1', 0.1)
    conv2 = separable_conv(64, 3, 'conv2', 0.1)
    conv3 = separable_conv(32, 5, 'conv3', 0.1)
    conv4 = separable_conv(64, 3, 'conv4', 0.1)
    conv5 = separable_conv(32, 5, 'conv5', 0.1)
    conv6 = separable_conv(64, 3, 'conv6', 0.1)
    conv7 = separable_conv(32, 5, 'conv7', 0.1)

    # Concatenate the outputs from the three groups
    concat_layer = Concatenate()(split_layers[1:])

    # Pass through Batch Normalization, Flatten, and Dense layers
    bn = BatchNormalization()(concat_layer)
    flat = Flatten()(bn)
    dense1 = Dense(units=512, activation='relu')(flat)
    dense2 = Dense(units=256, activation='relu')(dense1)
    dense3 = Dense(units=128, activation='relu')(dense2)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    # Construct the model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model