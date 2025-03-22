import keras
from keras.layers import Input, Lambda, Conv2D, Concatenate, Flatten, Dense

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))

        # Main Path
        x = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
        
        # Group 1: Remains unchanged
        group1 = x[0]

        # Group 2: Feature Extraction
        group2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x[1])

        # Group 3: Combined with Group 2
        group3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x[2])
        
        # Concatenate outputs
        main_path_output = Concatenate()([group1, group2, group3])

        # Branch Path
        branch_path_output = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)

        # Fuse Outputs
        fused_output = main_path_output + branch_path_output

        # Final Classification
        flatten_layer = Flatten()(fused_output)
        output_layer = Dense(units=10, activation='softmax')(flatten_layer)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model