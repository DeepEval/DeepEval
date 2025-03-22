import keras
from keras.layers import Input, Conv2D, BatchNormalization, Concatenate, Flatten, Dense, Reshape

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Define the first pathway
    def block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(BatchNormalization()(input_tensor))
        output_tensor = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
        return output_tensor
    
    pathway1 = block(input_layer)
    pathway1 = block(pathway1)
    pathway1 = block(pathway1)
    
    # Define the second pathway
    def block2(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(BatchNormalization()(input_tensor))
        output_tensor = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
        return output_tensor
    
    pathway2 = block2(input_layer)
    pathway2 = block2(pathway2)
    pathway2 = block2(pathway2)
    
    # Merge the outputs from both pathways
    merged_pathway = Concatenate()([pathway1, pathway2])
    
    # Flatten the merged pathway
    flatten_layer = Flatten()(merged_pathway)
    
    # Define the classification layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Create the Keras model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model