import keras
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first pathway
    pathway1 = input_layer

    # Define a repeated block structure for the first pathway
    def block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        output_tensor = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm)
        return output_tensor

    # Execute the block structure three times in the first pathway
    pathway1_output = block(pathway1)
    pathway1_output = block(pathway1_output)
    pathway1_output = block(pathway1_output)

    # Define the second pathway
    pathway2 = input_layer

    # Define a repeated block structure for the second pathway
    def block(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        output_tensor = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm)
        return output_tensor

    # Execute the block structure three times in the second pathway
    pathway2_output = block(pathway2)
    pathway2_output = block(pathway2_output)
    pathway2_output = block(pathway2_output)

    # Merge the outputs from both pathways
    merged_output = Concatenate()([pathway1_output, pathway2_output])

    # Flatten the merged output
    flatten_layer = Flatten()(merged_output)

    # Apply two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model