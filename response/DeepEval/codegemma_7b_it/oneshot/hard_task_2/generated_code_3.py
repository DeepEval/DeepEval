import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)

    # Define the main path
    main_path = []
    for group in x:
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        main_path.append(conv3)

    # Combine the outputs of the three groups using addition
    main_path = keras.layers.Add()(main_path)

    # Fuse the main path with the original input layer through addition
    fused_path = keras.layers.Add()([main_path, input_layer])

    # Flatten the combined features and feed them into the fully connected layer for classification
    flatten_layer = Flatten()(fused_path)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model