import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First Block: Main Path and Branch Path with Addition
    # Main Path
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    
    # Branch Path (Direct connection)
    branch_path = input_layer
    
    # Combine Main and Branch paths
    first_block_output = Add()([main_path, branch_path])
    
    # Second Block: Max Pooling layers with varying scales
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(first_block_output)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(first_block_output)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(first_block_output)
    
    # Flatten and Concatenate the outputs of pooling layers
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    
    second_block_output = Concatenate()([flatten1, flatten2, flatten3])
    
    # Fully Connected Layers
    dense1 = Dense(units=128, activation='relu')(second_block_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Build Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model