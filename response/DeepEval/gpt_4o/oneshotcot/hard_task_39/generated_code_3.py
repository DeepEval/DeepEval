import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Three max pooling layers with varying scales
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    
    # Flatten each pooling result
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)
    
    # Concatenate the flattened outputs
    concatenated = Concatenate()([flat1, flat2, flat3])
    
    # Fully connected layer and reshape to 4D tensor suitable for Block 2
    fc1 = Dense(units=128, activation='relu')(concatenated)
    reshaped = Reshape((4, 4, 8))(fc1)  # Example reshape, adjust dimensions as necessary
    
    # Block 2: Multiple branches for feature extraction
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshaped)
    branch3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshaped)
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshaped)
    
    # Concatenate the outputs from all branches
    block2_output = Concatenate()([branch1, branch2, branch3, branch4])
    
    # Final classification layer
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model