import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))

        # Block 1
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
        bn1 = BatchNormalization()(conv1)
        
        # Block 2
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(bn1)
        bn2 = BatchNormalization()(conv2)

        # Block 3
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(bn2)
        bn3 = BatchNormalization()(conv3)

        # Parallel Branch
        conv_branch = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
        bn_branch = BatchNormalization()(conv_branch)

        # Concatenate outputs
        merged = Concatenate()([bn3, bn_branch])
        
        flatten_layer = Flatten()(merged)
        dense1 = Dense(units=128, activation='relu')(flatten_layer)
        output_layer = Dense(units=10, activation='softmax')(dense1)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model