import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block1(input_tensor):
        split1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))(input_tensor)
        conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split1[0])
        conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(split1[1])
        conv1_3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(split1[2])
        concat = Concatenate(axis=-1)([conv1_1, conv1_2, conv1_3])
        return concat

    def block2(input_tensor):
        pool1 = MaxPooling2D(pool_size=(2, 2))(input_tensor)
        dense1 = Dense(units=128, activation='relu')(pool1)
        dense2 = Dense(units=64, activation='relu')(dense1)
        return dense2

    main_path = block1(input_layer)
    branch_path = block2(input_tensor=input_layer)
    fused_features = Add()([main_path, branch_path])
    output_layer = Dense(units=10, activation='softmax')(fused_features)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model