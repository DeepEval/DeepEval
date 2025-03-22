import keras
from keras.layers import Input, Conv2D, LayerNormalization, Dense, Add

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))
        
        # Depthwise Separable Convolutional Layer
        x = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding='same', depth_multiplier=1, activation='relu')(input_layer)
        x = LayerNormalization()(x)

        # Fully Connected Layers
        x = keras.layers.Reshape((x.shape[1] * x.shape[2], x.shape[3]))(x) 
        x = Dense(units=32, activation='relu')(x)
        x = Dense(units=32, activation='relu')(x)

        # Channel-wise Feature Transformation
        x = keras.layers.Reshape((x.shape[1], x.shape[2], 32))(x) 
        
        # Add Original Input
        output_layer = Add()([input_layer, x])
        
        output_layer = Dense(units=10, activation='softmax')(output_layer)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model