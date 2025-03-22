import keras
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Define the block structure
    def block(input_tensor):
        batch_norm = BatchNormalization()(input_tensor)
        relu = keras.layers.Activation('relu')(batch_norm)
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu)
        output_tensor = Concatenate()([input_tensor, conv])
        
        return output_tensor
    
    # Define the two pathways
    pathway1 = input_layer
    pathway2 = input_layer
    
    # Repeat the block structure three times for each pathway
    for i in range(3):
        pathway1 = block(pathway1)
        pathway2 = block(pathway2)

    # Merge the outputs from both pathways
    merged_output = Concatenate()([pathway1, pathway2])
    
    # Apply batch normalization
    batch_norm = BatchNormalization()(merged_output)
    
    # Flatten the output
    flatten_layer = Flatten()(batch_norm)
    
    # Define the fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model