import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Define a block that will be reused
    def block(input_tensor):
        # Step 2: Batch normalization
        batch_norm = BatchNormalization()(input_tensor)
        # Step 3: ReLU activation
        relu = ReLU()(batch_norm)
        # Step 4: 3x3 convolution
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(relu)
        # Step 5: Concatenate original input with the convolution output
        output_tensor = Concatenate()([input_tensor, conv])
        return output_tensor
    
    # Step 6: First pathway
    pathway1 = input_layer
    for _ in range(3):  # Repeat the block three times
        pathway1 = block(pathway1)
    
    # Step 7: Second pathway
    pathway2 = input_layer
    for _ in range(3):  # Repeat the block three times
        pathway2 = block(pathway2)
    
    # Step 8: Concatenate both pathways
    merged_output = Concatenate()([pathway1, pathway2])
    
    # Step 9: Flatten the result
    flatten_layer = Flatten()(merged_output)
    
    # Step 10: Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for MNIST

    # Step 11: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model