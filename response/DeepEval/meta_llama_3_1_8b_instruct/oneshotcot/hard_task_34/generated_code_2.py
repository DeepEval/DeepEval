import keras
from keras.layers import Input, SeparableConv2D, ReLU, Concatenate, Add, Conv2D, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    def block(input_tensor):
        # Define the separable convolutional layer
        conv_separable = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        # Define the ReLU activation function
        relu = ReLU()(conv_separable)
        # Concatenate the input and the convolved feature map along the channel dimension
        output_tensor = Concatenate()([input_tensor, relu])
        
        return output_tensor

    main_path = block(conv)
    main_path = block(main_path)
    main_path = block(main_path)
    
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(main_path)
    
    fusion = Add()([main_path, branch_path])
    bath_norm = keras.layers.BatchNormalization()(fusion)
    flatten_layer = Flatten()(bath_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model