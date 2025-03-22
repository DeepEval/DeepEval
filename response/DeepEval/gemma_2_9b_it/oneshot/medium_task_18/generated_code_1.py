import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():     

        input_layer = Input(shape=(32, 32, 3)) 

        # First convolutional block
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
        conv1_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
        conv1_5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(input_layer)
        block1_output = Concatenate()([conv1_1, conv1_3, conv1_5, pool1])

        # Second convolutional block
        conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_output)
        conv2_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_output)
        conv2_5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(block1_output)
        pool2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(block1_output)
        block2_output = Concatenate()([conv2_1, conv2_3, conv2_5, pool2])

        flatten_layer = Flatten()(block2_output)
        dense1 = Dense(units=128, activation='relu')(flatten_layer)
        output_layer = Dense(units=10, activation='softmax')(dense1)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model