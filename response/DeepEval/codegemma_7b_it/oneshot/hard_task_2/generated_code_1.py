import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_inputs = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)

    # Process each group of inputs sequentially
    features = []
    for split_input in split_inputs:
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_input)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        features.append(conv3)

    # Combine the outputs from the three groups using addition
    main_path = add(features)

    # Fuse the main path with the original input layer through addition
    merged = add([main_path, input_layer])

    # Flatten and feed into a fully connected layer for classification
    flatten_layer = Flatten()(merged)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model