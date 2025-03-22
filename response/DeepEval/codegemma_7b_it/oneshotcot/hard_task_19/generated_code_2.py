import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv2)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)

    def branch_path(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=256, activation='relu')(avg_pool)
        dense2 = Dense(units=128, activation='relu')(dense1)
        reshape = Reshape((1, 1, 128))(dense2)
        return reshape

    branch_output = branch_path(input_tensor=max_pooling)
    main_output = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(max_pooling)
    main_output = Flatten()(main_output)
    concat = Concatenate()([main_output, branch_output])
    dense3 = Dense(units=128, activation='relu')(concat)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model