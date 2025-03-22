import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))
        
        avg_pool = GlobalAveragePooling2D()(input_layer)

        # Two fully connected layers to learn channel correlations
        dense1 = Dense(units=3, activation='relu')(avg_pool)
        dense2 = Dense(units=3, activation='relu')(dense1)

        # Reshape weights to match input shape
        weights = Reshape((32, 32, 3))(dense2)

        # Element-wise multiplication with input feature map
        element_wise_product = Multiply()([input_layer, weights])
        
        # Flatten the result
        flatten = Flatten()(element_wise_product)
        
        # Final fully connected layer for classification
        output_layer = Dense(units=10, activation='softmax')(flatten)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model