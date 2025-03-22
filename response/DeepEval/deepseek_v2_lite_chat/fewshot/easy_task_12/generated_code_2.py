import keras
from keras.layers import Input, Conv2D, ReLU, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    def block1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=ReLU)(input_tensor)
        sep_conv1 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
        maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(sep_conv1)
        return maxpool1
    
    def block2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=ReLU)(input_tensor)
        sep_conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(sep_conv2)
        return maxpool2
    
    main_path = keras.Sequential([block1, block2])
    
    # Branch path
    branch_layer = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Sum the outputs of the main and branch paths
    merged = Add()([main_path.output, branch_layer])
    
    # Flatten the output
    flattened = Flatten()(merged)
    
    # Fully connected layer
    dense = Dense(units=128, activation='relu')(flattened)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model