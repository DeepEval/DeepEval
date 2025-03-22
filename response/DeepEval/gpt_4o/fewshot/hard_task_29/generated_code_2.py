import keras
from keras.layers import Input, Conv2D, Add, MaxPooling2D, Flatten, Concatenate, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # First Block
    # Main path with convolutional layers
    main_conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv1)
    
    # Branch path directly from input
    branch_path = input_layer
    
    # Combine main path and branch path
    block1_output = Add()([main_conv2, branch_path])
    
    # Second Block
    # Different scales of max pooling
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(block1_output)
    flatten1 = Flatten()(pool1)
    
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block1_output)
    flatten2 = Flatten()(pool2)
    
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(block1_output)
    flatten3 = Flatten()(pool3)
    
    # Concatenate flattened outputs
    block2_output = Concatenate()([flatten1, flatten2, flatten3])
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(block2_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model