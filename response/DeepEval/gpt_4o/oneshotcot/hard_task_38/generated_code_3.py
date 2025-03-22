import keras
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, Activation, Flatten, Dense

def block(input_tensor):
    # Apply batch normalization
    x = BatchNormalization()(input_tensor)
    # Apply ReLU activation
    x = Activation('relu')(x)
    # Apply 3x3 convolution
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    # Concatenate the original input with the new features
    output_tensor = Concatenate()([input_tensor, x])
    return output_tensor

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First pathway
    path1 = block(input_layer)
    path1 = block(path1)
    path1 = block(path1)
    
    # Second pathway
    path2 = block(input_layer)
    path2 = block(path2)
    path2 = block(path2)
    
    # Concatenate outputs of the two pathways
    merged_path = Concatenate()([path1, path2])
    
    # Flatten and add fully connected layers
    flatten_layer = Flatten()(merged_path)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model