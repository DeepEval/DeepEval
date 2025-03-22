import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First block
    def block1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
        
        # Second block
        def block2(input_tensor):
            conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
            pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
            return pool2
        
        pool2_output = block2(pool1)
        
        # Concatenate the outputs from both blocks
        concat_layer = Concatenate()([pool2_output, input_tensor])
        
        # Batch normalization and flattening
        batch_norm = BatchNormalization()(concat_layer)
        flatten = Flatten()(batch_norm)
        
        # Fully connected layers
        dense1 = Dense(units=128, activation='relu')(flatten)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        
        # Build the model
        model = keras.Model(inputs=input_layer, outputs=output_layer)
        return model
    
    # Construct the model
    model = block1(input_layer)
    model = block1(model)
    model = block2(model)
    model = block2(model)
    
    return model

# Build the model
model = dl_model()
model.summary()