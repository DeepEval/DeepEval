import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Concatenate, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels (RGB)

    def block(input_tensor):
        # Global Average Pooling
        pooled_output = GlobalAveragePooling2D()(input_tensor)
        
        # Fully connected layers to produce weights
        dense1 = Dense(units=input_tensor.shape[-1], activation='relu')(pooled_output)
        dense2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense1)  # Sigmoid for weights
        
        # Reshape to match input shape
        reshaped_weights = Dense(units=input_tensor.shape[1] * input_tensor.shape[2] * input_tensor.shape[-1], activation='sigmoid')(dense2)
        reshaped_weights = keras.layers.Reshape((input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[-1]))(reshaped_weights)
        
        # Element-wise multiplication
        scaled_output = Multiply()([input_tensor, reshaped_weights])
        return scaled_output

    # Create two branches using the same block
    branch1 = block(input_layer)
    branch2 = block(input_layer)

    # Concatenate the outputs of both branches
    concatenated = Concatenate()([branch1, branch2])

    # Flatten and apply a fully connected layer for classification
    flatten_layer = Flatten()(concatenated)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model