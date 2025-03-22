import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Lambda, Flatten, Dense, tf

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))
        
        # First Block: Dual-Path
        main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
        main_path = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
        main_path = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
        branch_path = input_layer
        
        combined = keras.layers.add([main_path, branch_path])

        # Second Block: Split-Path and Depthwise Separable Convolutions
        split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=2))(combined)
        
        # Group 1: 1x1 Conv
        conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
        
        # Group 2: 3x3 Conv
        conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])

        # Group 3: 5x5 Conv
        conv3_1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layer[2])
        
        concat_output = Concatenate(axis=2)([conv1_1, conv2_1, conv3_1])
        
        # Flatten and Fully Connected Layers
        flatten_layer = Flatten()(concat_output)
        dense1 = Dense(units=128, activation='relu')(flatten_layer)
        output_layer = Dense(units=10, activation='softmax')(dense1)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model