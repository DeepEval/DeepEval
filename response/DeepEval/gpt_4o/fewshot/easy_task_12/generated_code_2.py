import keras
from keras.layers import Input, SeparableConv2D, MaxPooling2D, Conv2D, Add, Flatten, Dense, ReLU

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Main Path
    def main_path(input_tensor):
        # First Block
        relu1 = ReLU()(input_tensor)
        sep_conv1 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu1)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(sep_conv1)
        
        # Second Block
        relu2 = ReLU()(pool1)
        sep_conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu2)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(sep_conv2)
        
        return pool2
    
    main_path_output = main_path(input_layer)
    
    # Branch Path
    branch_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)

    # Adding Outputs
    adding_layer = Add()([main_path_output, branch_conv])
    
    # Fully Connected Layers
    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model