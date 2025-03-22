import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input along the channel dimension into three groups
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)

    # Sequentially pass each group through a series of convolutions
    conv1_layer = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(split_layer[0])
    conv2_layer = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(split_layer[1])
    conv3_layer = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(split_layer[2])

    # Add the outputs from each group to form the main path
    main_path = conv1_layer + conv2_layer + conv3_layer

    # Fuse the main path with the original input layer through another addition
    fusion_layer = main_path + input_layer

    # Flatten the combined features
    flatten_layer = Flatten()(fusion_layer)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model