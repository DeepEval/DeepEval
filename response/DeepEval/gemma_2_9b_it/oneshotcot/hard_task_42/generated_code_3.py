import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Concatenate, Dense, Reshape

def dl_model():     

        input_layer = Input(shape=(28, 28, 1))

        # Block 1
        path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
        path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
        path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)

        flat1 = Flatten()(path1)
        flat2 = Flatten()(path2)
        flat3 = Flatten()(path3)

        drop1 = Dropout(0.25)(flat1)
        drop2 = Dropout(0.25)(flat2)
        drop3 = Dropout(0.25)(flat3)

        concat_block1 = Concatenate()([drop1, drop2, drop3])

        # Fully connected layer and reshape
        dense1 = Dense(units=128, activation='relu')(concat_block1)
        reshape_layer = Reshape((1, 128))(dense1)

        # Block 2
        path4_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
        path4_2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
        path4_2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path4_2)
        path4_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
        path4_3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path4_3)
        path4_3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path4_3)
        path4_4 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(reshape_layer)
        path4_4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4_4)

        concat_block2 = Concatenate()([path4_1, path4_2, path4_3, path4_4])

        # Final classification layers
        flatten_layer = Flatten()(concat_block2)
        dense2 = Dense(units=64, activation='relu')(flatten_layer)
        output_layer = Dense(units=10, activation='softmax')(dense2)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model