import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Reshape
from keras.layers import concatenate as concat

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    block1_output = block1(input_layer)
    block2_input = Reshape((64,))(block1_output)  # reshape the output to 4 dimensions
    block2_input = Dense(units=128, activation='relu')(block2_input)
    block2_output = block2(block2_input)
    output_layer = Dense(units=10, activation='softmax')(block2_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

def block1(input_tensor):

    path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
    path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
    path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)

    output_tensor = Concatenate()([path1, path2, path3])
    output_tensor = Flatten()(output_tensor)
    output_tensor = Dropout(0.2)(output_tensor)
    output_tensor = Dense(units=32, activation='relu')(output_tensor)
    output_tensor = Dropout(0.2)(output_tensor)

    return output_tensor

def block2(input_tensor):

    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    path2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)

    path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path3)

    path4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
    path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)

    output_tensor = Concatenate()([path1, path2, path3, path4])
    output_tensor = Flatten()(output_tensor)
    output_tensor = Dense(units=64, activation='relu')(output_tensor)

    return output_tensor