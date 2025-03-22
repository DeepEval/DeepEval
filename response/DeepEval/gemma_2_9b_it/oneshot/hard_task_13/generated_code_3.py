import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Reshape

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    def block1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_tensor)
        output_tensor = Concatenate()([conv1, conv2, conv3, maxpool])
        return output_tensor

    block1_output = block1(input_tensor=input_layer)

    def block2(input_tensor):
        pooled_features = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=input_tensor.shape[-1], activation='relu')(pooled_features)
        dense2 = Dense(units=input_tensor.shape[-1], activation='relu')(dense1)
        reshape_layer = Reshape(target_shape=(input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[-1]))(dense2)
        output_tensor = keras.layers.Multiply()([input_tensor, reshape_layer])
        return output_tensor

    block2_output = block2(input_tensor=block1_output)

    output_layer = Dense(units=10, activation='softmax')(block2_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model