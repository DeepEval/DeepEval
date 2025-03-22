import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Conv2DTranspose, Flatten, Dense

def dl_model():
    
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: Two blocks of convolution followed by average pooling
    path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
    path1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(path1)
    
    # Path 2: Single convolutional layer
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the outputs of both pathways
    output_tensor = Add()([path1, path2])
    
    # Flatten the output
    flatten_layer = Flatten()(output_tensor)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model