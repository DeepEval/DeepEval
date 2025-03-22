import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():     
    
    input_layer = Input(shape=(32, 32, 3)) 

    # Branch 1
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch1)

    def block(input_tensor):
        pooled = GlobalAveragePooling2D()(input_tensor)
        fc1 = Dense(units=input_tensor.shape[-1], activation='relu')(pooled)
        fc2 = Dense(units=input_tensor.shape[-1], activation='relu')(fc1)
        reshaped = Reshape((input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[-1]))(fc2)
        return input_tensor * reshaped 

    branch1 = block(branch1)

    # Branch 2
    branch2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch2)
    branch2 = block(branch2)

    # Concatenate branches
    merged = Concatenate()([branch1, branch2])
    
    flatten_layer = Flatten()(merged)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model