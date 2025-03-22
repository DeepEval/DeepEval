import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense, Reshape

def dl_model():     

        input_layer = Input(shape=(28, 28, 1))
        
        # Block 1
        pool1_1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
        pool1_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
        pool1_3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
        
        flat1_1 = Flatten()(pool1_1)
        flat1_2 = Flatten()(pool1_2)
        flat1_3 = Flatten()(pool1_3)
        
        drop1_1 = Dropout(0.25)(flat1_1)
        drop1_2 = Dropout(0.25)(flat1_2)
        drop1_3 = Dropout(0.25)(flat1_3)

        concat_block1 = Concatenate()([drop1_1, drop1_2, drop1_3])
        
        dense1 = Dense(units=128, activation='relu')(concat_block1)
        reshape_layer = Reshape((1, 128))(dense1)
        
        # Block 2
        conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
        conv2_2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
        conv2_2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv2_2)
        
        conv2_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
        conv2_3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv2_3)
        conv2_3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv2_3)
        
        conv2_4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(reshape_layer)
        conv2_4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2_4)

        concat_block2 = Concatenate(axis=3)([conv2_1, conv2_2, conv2_3, conv2_4])

        flatten_block2 = Flatten()(concat_block2)
        dense2 = Dense(units=64, activation='relu')(flatten_block2)
        output_layer = Dense(units=10, activation='softmax')(dense2)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model