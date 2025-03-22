import keras
from keras.layers import Input, Conv2D, Add, MaxPooling2D, Flatten, Concatenate, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))

    # Step 2: First Block
    # Main Path
    conv1_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_main = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_main)
    
    # Branch Path
    branch_path = input_layer
    
    # Adding the outputs from both paths
    block1_output = Add()([conv2_main, branch_path])

    # Step 3: Second Block
    # Max Pooling Layers
    max_pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(block1_output)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block1_output)
    max_pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(block1_output)

    # Flattening the outputs
    flatten1 = Flatten()(max_pool1)
    flatten2 = Flatten()(max_pool2)
    flatten3 = Flatten()(max_pool3)

    # Concatenating the flattened outputs
    block2_output = Concatenate()([flatten1, flatten2, flatten3])

    # Step 4: Fully Connected Layers
    dense1 = Dense(units=128, activation='relu')(block2_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Building the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model