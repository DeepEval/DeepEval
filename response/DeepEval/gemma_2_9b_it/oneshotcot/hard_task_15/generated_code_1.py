import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))

        # Main Path
        x = GlobalAveragePooling2D()(input_layer)
        x = Dense(units=32, activation='relu')(x)
        x = Dense(units=32, activation='relu')(x)
        x = Reshape(target_shape=(32, 32, 3))(x)  
        x = keras.layers.multiply([input_layer, x])

        # Branch Path
        branch_path = input_layer

        # Combine outputs
        merged = Concatenate()([x, branch_path])

        # Final Classification Layers
        merged = Flatten()(merged)
        merged = Dense(units=128, activation='relu')(merged)
        output_layer = Dense(units=10, activation='softmax')(merged)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model