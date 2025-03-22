import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: Features extraction block
    def features_extraction_block(input_tensor):
        # Block 1
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(conv1)
        # Block 2
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(conv2)
        return pool2  # This is the output of the features extraction block
    
    # Path 2: Local detail extraction block
    def local_detail_extraction_block(input_tensor):
        # Single convolutional layer
        conv = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(input_tensor)
        return conv  # This is the output of the local detail extraction block
    
    # Extract features
    features_extraction_output = features_extraction_block(input_layer)
    local_detail_output = local_detail_extraction_block(input_layer)
    
    # Add the outputs from both paths
    added_output = Add()([features_extraction_output, local_detail_output])
    
    # Flatten and fully connected layers
    flatten = Flatten()(added_output)
    dense = Dense(units=128, activation='relu')(flatten)
    
    # Output layer
    output = Dense(units=10, activation='softmax')(dense)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Instantiate and return the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()