import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def branch_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_tensor)
        output_tensor = Concatenate()([conv1, conv2, conv3, maxpool])
        return output_tensor

    block1_output = branch_block(input_tensor=input_layer)

    block2_output = GlobalAveragePooling2D()(block1_output)
    dense1 = Dense(units=block2_output.shape[-1], activation='relu')(block2_output)
    dense2 = Dense(units=block2_output.shape[-1], activation='relu')(dense1)
    reshape_layer = Reshape(target_shape=block1_output.shape[1:3] + (block2_output.shape[-1],))(dense2)
    weighted_output = input_layer * reshape_layer

    output_layer = Dense(units=10, activation='softmax')(weighted_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model