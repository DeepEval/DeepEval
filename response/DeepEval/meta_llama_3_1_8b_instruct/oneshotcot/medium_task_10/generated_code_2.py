import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, Activation, AveragePooling2D, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=None)(input_layer)
    batch_norm = BatchNormalization()(conv)
    relu = Activation('relu')(batch_norm)

    def basic_block(input_tensor):

        residual = input_tensor

        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_tensor)
        batch_norm = BatchNormalization()(conv)
        relu = Activation('relu')(batch_norm)
        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(relu)
        batch_norm = BatchNormalization()(conv)

        output_tensor = Add()([batch_norm, residual])
        output_tensor = Activation('relu')(output_tensor)

        return output_tensor
    
    def second_level_residual_block(input_tensor):

        residual = input_tensor

        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_tensor)
        batch_norm = BatchNormalization()(conv)
        relu = Activation('relu')(batch_norm)
        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(relu)
        batch_norm = BatchNormalization()(conv)
        independent_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_tensor)
        independent_batch_norm = BatchNormalization()(independent_conv)
        independent_relu = Activation('relu')(independent_batch_norm)
        independent_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(independent_relu)

        output_tensor = Add()([batch_norm, independent_conv])
        output_tensor = Activation('relu')(output_tensor)

        return output_tensor
    
    first_level_residual = basic_block(input_tensor=relu)
    second_level_residual = second_level_residual_block(input_tensor=first_level_residual)
    second_level_residual = second_level_residual_block(input_tensor=second_level_residual)

    global_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=None)(relu)
    global_batch_norm = BatchNormalization()(global_branch)

    final_output = Add()([second_level_residual, global_batch_norm])

    average_pooling = AveragePooling2D(pool_size=(8, 8))(final_output)
    flatten_layer = Flatten()(average_pooling)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model