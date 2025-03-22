import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Conv2D, MaxPooling2D, Reshape, Multiply
    
def dl_model():     

        input_layer = Input(shape=(32, 32, 3))  

        # Block 1
        x = GlobalAveragePooling2D()(input_layer)
        x = Dense(64, activation='relu')(x)  
        x = Dense(32, activation='relu')(x)  
        x = Reshape((32, 32, 3))(x)  
        x_weighted = Multiply()([input_layer, x])

        # Block 2
        y = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
        y = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(y)
        y = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(y)

        # Fusion
        combined = keras.layers.add([x_weighted, y]) 

        # Output Layers
        out = Dense(128, activation='relu')(combined)
        output_layer = Dense(10, activation='softmax')(out)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model