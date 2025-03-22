import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    channel_groups = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)

    # Define a series of convolutions for each group
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(channel_groups[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(channel_groups[1])
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(channel_groups[2])

    # Combine the outputs from each group using addition
    main_path = Concatenate()([conv1, conv2, conv3])

    # Fuse the main path with the original input layer using addition
    fused_input = Concatenate()([input_layer, main_path])

    # Flatten the combined features and feed them into a fully connected layer for classification
    flattened = Flatten()(fused_input)
    dense = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model