import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model(): 
    input_layer = Input(shape=(28, 28, 1))

    # First Block
    main_path = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(main_path) 
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer) 
    block_output = Add()([main_path, branch_path])

    # Second Block
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(block_output)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block_output)
    pool4 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(block_output)
    
    flattened_outputs = [Flatten()(pool1), Flatten()(pool2), Flatten()(pool4)]
    block_output = Concatenate()(flattened_outputs)

    # Final Layers
    dense1 = Dense(units=128, activation='relu')(block_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model