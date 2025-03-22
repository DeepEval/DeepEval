import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))
        
        # Branch 1: 1x1 convolution for dimensionality reduction
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

        # Branch 2: 1x1 + 3x3 convolutions
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

        # Branch 3: 1x1 + 5x5 convolutions
        branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
        branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch3)

        # Branch 4: 3x3 max pooling + 1x1 convolution
        branch4 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(input_layer)
        branch4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)

        # Concatenate the outputs of all branches
        output_branches = Concatenate()([branch1, branch2, branch3, branch4])

        # Batch normalization
        output_branches = BatchNormalization()(output_branches)

        # Flatten the features
        flatten_layer = Flatten()(output_branches)

        # Dense layers
        dense1 = Dense(units=128, activation='relu')(flatten_layer)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model