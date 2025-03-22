from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    def basic_block(input_tensor, filters):
        # Main path
        conv_main = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(input_tensor)
        bn_main = BatchNormalization()(conv_main)
        relu_main = ReLU()(bn_main)
        
        # Branch path
        conv_branch = Conv2D(filters=filters, kernel_size=(1, 1), padding='same')(input_tensor)
        
        # Feature fusion
        output_tensor = Add()([relu_main, conv_branch])
        
        return output_tensor

    # Input layer for CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer to reduce dimensionality
    conv_initial = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Two consecutive basic blocks
    block1_output = basic_block(conv_initial, 16)
    block2_output = basic_block(block1_output, 16)
    
    # Average pooling layer
    avg_pooling = AveragePooling2D(pool_size=(2, 2))(block2_output)
    
    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(avg_pooling)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model