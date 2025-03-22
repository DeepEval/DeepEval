import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    def repeated_block(input_tensor, num_filters):
        # Batch normalization and ReLU activation
        batch_norm = BatchNormalization()(input_tensor)
        relu = keras.activations.relu(batch_norm)
        
        # 3x3 convolutional layer
        conv = Conv2D(num_filters, kernel_size=3, padding='same')(relu)
        
        # MaxPooling2D for spatial downsampling
        maxpooling = MaxPooling2D(pool_size=(2, 2))(conv)
        
        return maxpooling

    input_layer = Input(shape=(28, 28, 1))

    # First pathway
    pathway1_output = repeated_block(input_tensor=input_layer, num_filters=32)
    for _ in range(3):  # Execute the block three times
        pathway1_output = repeated_block(input_tensor=pathway1_output, num_filters=32)
    pathway1_output = Concatenate(axis=-1)([pathway1_output, input_layer])

    # Second pathway
    pathway2_output = repeated_block(input_tensor=input_layer, num_filters=64)
    for _ in range(3):  # Execute the block three times
        pathway2_output = repeated_block(input_tensor=pathway2_output, num_filters=64)
    pathway2_output = Concatenate(axis=-1)([pathway2_output, input_layer])

    # Concatenate and classify
    concatenated_output = Concatenate(axis=-1)([pathway1_output, pathway2_output])
    dense1 = Dense(units=128, activation='relu')(concatenated_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model