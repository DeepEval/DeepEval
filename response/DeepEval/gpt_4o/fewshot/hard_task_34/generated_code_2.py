import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Add, Flatten, Dense, Concatenate, Activation

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    def specific_block(input_tensor):
        relu = Activation('relu')(input_tensor)
        separable_conv = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu)
        output_tensor = Concatenate(axis=-1)([input_tensor, separable_conv])
        return output_tensor
    
    # Main path with three specific blocks
    main_path = specific_block(input_layer)
    main_path = specific_block(main_path)
    main_path = specific_block(main_path)
    
    # Branch path with a convolutional layer
    branch_path = Conv2D(filters=main_path.shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    
    # Fuse features from both paths
    fused_path = Add()([main_path, branch_path])
    
    # Flatten and fully connected layer for classification
    flatten = Flatten()(fused_path)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model