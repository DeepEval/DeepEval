import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow import split

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    main_path = Lambda(lambda x: x[:, :, :, 0:1])(input_layer)
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block(input_tensor):

        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        return conv1
    
    block_output = block(input_tensor=main_path)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block_output)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_path)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_path)

    branch_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_path)
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_path)
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)

    branch_path = Lambda(lambda x: x[:, :, :, 0:1])(branch_path)

    concat_path = Concatenate()([main_path, branch_path])
    concat_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat_path)

    flatten_layer = Flatten()(concat_path)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model