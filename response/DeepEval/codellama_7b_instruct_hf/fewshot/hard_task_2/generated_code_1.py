import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Splits the input into three groups along the channel dimension
    splits = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    # Each group will sequentially pass through a series of convolutions
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(splits[0])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(splits[1])
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(splits[2])

    # Combines the outputs from the three groups using an addition operation
    main_path = Add()([conv1, conv2, conv3])

    # Fuses the main path with the original input layer through another addition
    adding_layer = Add()([main_path, input_layer])

    # Flattens the combined features into a one-dimensional vector
    flatten_layer = Flatten()(adding_layer)

    # Feeds the flattened features into a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Creates and returns the constructed model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model