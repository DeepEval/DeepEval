import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Concatenate, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor):
        # Global average pooling
        gap = GlobalAveragePooling2D()(input_tensor)
        
        # First fully connected layer to produce weights
        fc1 = Dense(units=input_tensor.shape[-1] // 2, activation='relu')(gap)
        
        # Second fully connected layer to produce weights
        fc2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(fc1)
        
        # Reshape to match input dimensions (1, 1, channels)
        scale = keras.layers.Reshape((1, 1, input_tensor.shape[-1]))(fc2)
        
        # Element-wise multiplication
        scaled_output = Multiply()([input_tensor, scale])
        
        return scaled_output

    # Branch 1
    branch1 = block(input_layer)

    # Branch 2
    branch2 = block(input_layer)

    # Concatenate outputs of both branches
    concatenated = Concatenate()([branch1, branch2])

    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(concatenated)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# To create the model
model = dl_model()
model.summary()