import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split input into three groups along channel dimension
    input_groups = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)

    # Convolutions for each group
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[0])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_groups[1])
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[2])

    # Combine outputs from convolutions
    main_path = tf.add(conv1, conv2, conv3)

    # Fuse main path with original input
    fused_input = tf.add(input_layer, main_path)

    # Flatten and pass through fully connected layers
    flattened = Flatten()(fused_input)
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Define model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model