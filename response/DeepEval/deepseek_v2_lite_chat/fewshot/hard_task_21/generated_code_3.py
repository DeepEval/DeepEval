import keras
from keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, Add, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def extract_features(input_tensor):
        split1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))(input_tensor)
        conv1_1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split1[0])
        conv1_2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split1[1])
        conv1_3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split1[2])
        concat_main = Concatenate()([conv1_1, conv1_2, conv1_3])
        return concat_main

    main_path_output = extract_features(input_tensor=input_layer)

    # Branch path
    def branch_features(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv1

    branch_path_output = branch_features(input_tensor=input_layer)
    branch_path_output = Flatten()(branch_path_output)

    # Add main path and branch path outputs
    add_layer = Add()([main_path_output, branch_path_output])

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(add_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model