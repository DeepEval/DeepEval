import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    def main_path(input_tensor):
        # First convolutional layer
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_tensor)
        # Increase the feature map width
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
        # Restore the number of channels
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv2)
        
        # Branch path
        branch_input = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_tensor)
        
        # Combine the paths using an addition operation
        combined = Add()([conv3, branch_input])
        
        # Batch normalization and flattening
        batch_norm = BatchNormalization()(combined)
        flatten_layer = Flatten()(batch_norm)
        
        # Fully connected layers for classification
        dense1 = Dense(units=128, activation='relu')(flatten_layer)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        
        model = keras.Model(inputs=input_layer, outputs=output_layer)
        return model
    
    # Construct the model
    model = main_path(input_layer)
    
    return model

# Instantiate and print the constructed model
model = dl_model()
model.summary()