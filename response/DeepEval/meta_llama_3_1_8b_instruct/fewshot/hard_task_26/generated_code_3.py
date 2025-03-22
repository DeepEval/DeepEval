import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Concatenate
from keras.regularizers import l2

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def main_path(input_tensor):
        initial_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input_tensor)
        
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(initial_conv)
        
        branch2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(initial_conv)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(branch2)
        branch2 = UpSampling2D(size=(2, 2))(branch2)

        branch3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(initial_conv)
        branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(branch3)
        branch3 = UpSampling2D(size=(2, 2))(branch3)

        output_tensor = Concatenate()([branch1, branch2, branch3])
        output_tensor = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(output_tensor)

        return output_tensor
    
    def branch_path(input_tensor):
        initial_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input_tensor)
        return initial_conv
    
    main_path_output = main_path(input_layer)
    branch_path_output = branch_path(input_layer)
    adding_layer = Add()([main_path_output, branch_path_output])
    
    flatten_layer = Flatten()(adding_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model