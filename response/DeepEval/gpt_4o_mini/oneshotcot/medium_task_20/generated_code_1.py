import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))

    # Step 4: Define a block
    def block(input_tensor):
        # Step 4.1: First path - 1x1 convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Step 4.2: Second path - 1x1 followed by two 3x3 convolutions
        path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        
        # Step 4.3: Third path - 1x1 followed by a 3x3 convolution
        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        
        # Step 4.4: Fourth path - max pooling followed by 1x1 convolution
        path4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        
        # Step 4.5: Concatenate the outputs of the paths
        output_tensor = Concatenate()([path1, path2, path3, path4])

        return output_tensor
    
    # Create the block output
    block_output = block(input_tensor=input_layer)
    
    # Step 5: Add flatten layer
    flatten_layer = Flatten()(block_output)
    
    # Step 6: Add dense layer with 128 units
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    
    # Step 7: Add output layer with softmax activation for 10 classes
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()