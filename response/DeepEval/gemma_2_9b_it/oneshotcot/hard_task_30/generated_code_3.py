import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, tf

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))

        # First Block: Dual-Path
        main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
        main_path = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
        main_path = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path) 

        branch_path = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

        output_block1 = keras.layers.Add()([main_path, branch_path])

        # Second Block: Split-Path with Depthwise Separable Convolutions
        split_tensor = Lambda(lambda x: tf.split(x, 3, axis=2))(output_block1)

        group1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
        group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
        group3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])

        concatenated_output = Concatenate()( [group1, group2, group3] )

        batch_norm = BatchNormalization()(concatenated_output)
        flatten_layer = Flatten()(batch_norm)
        dense1 = Dense(units=128, activation='relu')(flatten_layer)
        output_layer = Dense(units=10, activation='softmax')(dense1)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model