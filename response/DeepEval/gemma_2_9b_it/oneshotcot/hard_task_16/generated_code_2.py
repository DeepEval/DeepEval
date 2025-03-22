import keras
from keras.layers import Input, Conv2D, Lambda, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3)) 

    # Block 1
    def block1(input_tensor):
        split_tensor = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
        path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path1)

        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)

        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path3)

        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor

    block1_output = block1(input_layer)
    transition_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_output)

    # Block 2
    def block2(input_tensor):
        pool_output = MaxPooling2D(pool_size=(7, 7), strides=(7, 7))(transition_conv)
        
        dense1 = Dense(units=64, activation='relu')(Flatten()(pool_output))
        dense2 = Dense(units=64, activation='relu')(dense1)
        weights = Reshape((32 * 32 * 64,))(dense2)
        
        reshaped_weights = Reshape((32, 32, 64))(weights)
        output_tensor = input_tensor * reshaped_weights 
        return output_tensor

    main_path_output = block2(transition_conv)

    # Branch
    branch_output = input_layer

    # Combine outputs and final classification
    combined_output = main_path_output + branch_output
    output_layer = Dense(units=10, activation='softmax')(Flatten()(combined_output))

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model