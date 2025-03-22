import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Multiply

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))
        # ... (Rest of your model code here) ...
        
        x = GlobalAveragePooling2D()(input_layer)
        
        # Fully connected layers for channel correlation learning
        dense1 = Dense(units=3, activation='relu')(x)  
        dense2 = Dense(units=3, activation='relu')(dense1)

        # Reshape weights to match input shape
        reshape_layer = Reshape((32, 32, 3))(dense2) 

        # Element-wise multiplication
        output = Multiply()([input_layer, reshape_layer])

        # Flatten and final classification layer
        flatten_layer = Flatten()(output)
        output_layer = Dense(units=10, activation='softmax')(flatten_layer)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model