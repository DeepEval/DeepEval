import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Add, Concatenate, Flatten, Dense, Activation

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    def feature_block(input_tensor):
        # ReLU activation followed by Separable Convolutional layer
        relu = Activation('relu')(input_tensor)
        separable_conv = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(relu)
        
        # Concatenate input with convolved feature map
        output_tensor = Concatenate(axis=-1)([input_tensor, separable_conv])
        return output_tensor

    # Main path with three sequential feature blocks
    main_path = feature_block(input_layer)
    main_path = feature_block(main_path)
    main_path = feature_block(main_path)

    # Branch path with a convolutional layer matching main path's channels
    branch_path = Conv2D(filters=main_path.shape[-1], kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse main path and branch path through addition
    fused_path = Add()([main_path, branch_path])

    # Flatten and output fully connected layer for classification
    flatten = Flatten()(fused_path)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model