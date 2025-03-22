import keras
from keras.layers import Input, Conv2D, Dropout, Lambda, Concatenate, BatchNormalization, Flatten, Dense, tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    x_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x_main = Dropout(0.2)(x_main)
    x_main = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x_main) 

    x_branch = input_layer

    x_combined = keras.layers.add([x_main, x_branch])

    # Block 2
    def split_and_process(input_tensor):
        split_tensor = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)  
        group1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
        group1 = Dropout(0.2)(group1)
        group2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
        group2 = Dropout(0.2)(group2)
        group3 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
        group3 = Dropout(0.2)(group3)
        return Concatenate()([group1, group2, group3]) 

    x_block2 = split_and_process(x_combined)

    # Final Layers
    x_flatten = Flatten()(x_block2)
    output_layer = Dense(units=10, activation='softmax')(x_flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model