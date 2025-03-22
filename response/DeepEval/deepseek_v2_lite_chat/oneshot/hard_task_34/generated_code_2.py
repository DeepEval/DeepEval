import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    def main_path(input_tensor):
        # Block 1
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Block 2
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        # Block 3
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        # Pooling layer
        pool = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        # Branch path
        branch_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        concat = Concatenate(axis=-1)([pool, branch_conv])
        
        return concat
    
    # Output of the main path
    main_output = main_path(input_layer)
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(main_output)
    flatten = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()