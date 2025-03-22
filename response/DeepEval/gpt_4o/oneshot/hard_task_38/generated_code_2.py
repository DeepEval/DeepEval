import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # First BatchNorm and ReLU
        x = BatchNormalization()(input_tensor)
        x = ReLU()(x)
        
        # 3x3 Convolution
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        
        # Concatenate original input with new features
        output_tensor = Concatenate()([input_tensor, x])
        
        return output_tensor

    # Define two pathways
    path1 = input_layer
    path2 = input_layer
    
    # Each pathway executes the block three times
    for _ in range(3):
        path1 = block(path1)
        path2 = block(path2)
    
    # Concatenate outputs from both pathways
    merged = Concatenate()([path1, path2])
    
    # Flatten and fully connected layers for classification
    flatten_layer = Flatten()(merged)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model