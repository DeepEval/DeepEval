import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Average pooling layers with different strides
    avg_pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    avg_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    avg_pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    
    # Flatten the outputs of the pooling layers and concatenate them
    concatenated = Concatenate()(
        [
            avg_pool1,
            avg_pool2,
            avg_pool3
        ]
    )
    
    # Flatten the concatenated layer
    flattened = Flatten()(concatenated)
    
    # Two dense layers for processing the flattened output
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer with 10 classes (CIFAR-10 has 10 classes)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()