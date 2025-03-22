import keras
from keras.layers import Input, MaxPooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D
    
def dl_model():     

        input_layer = Input(shape=(28, 28, 1))
        
        # Block 1: Max Pooling Layers
        pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
        pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

        flat1 = Flatten()(pool1)
        flat2 = Flatten()(pool2)
        flat3 = Flatten()(pool3)

        concat_block1 = Concatenate()([flat1, flat2, flat3])

        # Dense Layer and Reshape
        dense1 = Dense(units=128, activation='relu')(concat_block1)
        reshape_layer = Reshape((1, 128))(dense1) 

        # Block 2: Feature Extraction Branches
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
        pool4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(reshape_layer) 

        concat_block2 = Concatenate()([conv1, conv2, conv3, pool4])

        # Final Layers
        flat_final = Flatten()(concat_block2)
        output_layer = Dense(units=10, activation='softmax')(flat_final)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model