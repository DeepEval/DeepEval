import keras
from keras.layers import Input, MaxPooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D

def dl_model():     

        input_layer = Input(shape=(28, 28, 1))
        
        # Block 1: Multi-scale Max Pooling
        pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
        pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
        
        flat1 = Flatten()(pool1)
        flat2 = Flatten()(pool2)
        flat3 = Flatten()(pool3)
        
        concat1 = Concatenate()([flat1, flat2, flat3])
        
        # FC layer and reshape for Block 2
        dense1 = Dense(units=128, activation='relu')(concat1)
        reshape_layer = Reshape((4, 32))(dense1) 

        # Block 2: Feature Extraction Branches
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
        branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
        branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshape_layer)

        concat2 = Concatenate()([branch1, branch2, branch3, branch4])

        # Final Classification
        flatten_final = Flatten()(concat2)
        output_layer = Dense(units=10, activation='softmax')(flatten_final)
        
        model = keras.Model(inputs=input_layer, outputs=output_layer)
        
        return model