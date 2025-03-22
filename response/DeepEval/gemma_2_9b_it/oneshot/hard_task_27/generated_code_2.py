import keras
from keras.layers import Input, Conv2D, LayerNormalization, Dense, Concatenate

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))
        
        # Depthwise separable convolution with layer normalization
        depthwise_conv = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), 
                               padding='same', depth_multiplier=1, activation='relu')(input_layer)
        layer_norm = LayerNormalization()(depthwise_conv)

        # Two fully connected layers for channel-wise feature transformation
        fc1 = Dense(units=32, activation='relu')(layer_norm)
        fc2 = Dense(units=32, activation='relu')(fc1)

        # Combine original input and processed features
        combined_features = Concatenate()([input_layer, fc2])

        # Final classification layers
        dense3 = Dense(units=128, activation='relu')(combined_features)
        output_layer = Dense(units=10, activation='softmax')(dense3)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model