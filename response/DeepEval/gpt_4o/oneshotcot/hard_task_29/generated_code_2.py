import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First Block
    # Main Path
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    
    # Branch Path (Direct connection from input)
    branch_path = input_layer
    
    # Adding main path and branch path
    block1_output = Add()([main_path, branch_path])
    
    # Second Block
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(block1_output)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block1_output)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(block1_output)
    
    # Flatten each pooled output
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)
    
    # Concatenate flattened outputs
    block2_output = Concatenate()([flat1, flat2, flat3])
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(block2_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model