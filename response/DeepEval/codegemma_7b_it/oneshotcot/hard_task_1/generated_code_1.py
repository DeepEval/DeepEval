import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Activation, Multiply, Reshape, BatchNormalization, Dropout, Conv2DTranspose, concatenate, add

def block1(input_tensor, filters):
    path1 = GlobalAveragePooling2D()(input_tensor)
    path1 = Dense(filters=filters, activation='relu')(path1)
    path1 = Dense(filters=filters, activation='relu')(path1)

    path2 = GlobalMaxPooling2D()(input_tensor)
    path2 = Dense(filters=filters, activation='relu')(path2)
    path2 = Dense(filters=filters, activation='relu')(path2)

    output_tensor = add([path1, path2])
    output_tensor = Activation('sigmoid')(output_tensor)
    output_tensor = Reshape((input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))(output_tensor)
    output_tensor = Multiply()([input_tensor, output_tensor])

    return output_tensor

def block2(input_tensor, filters):
    path1 = AveragePooling2D()(input_tensor)
    path1 = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path1)

    path2 = MaxPooling2D()(input_tensor)
    path2 = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)

    output_tensor = concatenate([path1, path2], axis=3)
    output_tensor = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_tensor)
    output_tensor = Activation('sigmoid')(output_tensor)
    output_tensor = Multiply()([input_tensor, output_tensor])

    return output_tensor

def block3(input_tensor):
    output_tensor = Conv2D(filters=input_tensor.shape[3], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

    return output_tensor

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    block1_output = block1(input_tensor=conv_layer, filters=32)
    block2_output = block2(input_tensor=block1_output, filters=32)
    block3_output = block3(input_tensor=block2_output)

    flattened_output = Flatten()(block3_output)
    dense_layer = Dense(units=10, activation='softmax')(flattened_output)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model