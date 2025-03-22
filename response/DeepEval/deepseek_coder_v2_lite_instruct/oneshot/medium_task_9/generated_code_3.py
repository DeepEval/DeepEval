import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer to reduce dimensionality
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    relu1 = ReLU()(batch_norm1)

    # Basic block definition
    def basic_block(input_tensor):
        # Main path
        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        relu = ReLU()(batch_norm)
        
        # Branch path
        branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        batch_norm_branch = BatchNormalization()(branch)
        relu_branch = ReLU()(batch_norm_branch)
        
        # Feature fusion by addition
        output_tensor = Add()([relu, relu_branch])
        return output_tensor

    # First block
    block1_output = basic_block(relu1)
    
    # Second block
    block2_output = basic_block(block1_output)
    
    # Average pooling layer
    avg_pool = AveragePooling2D(pool_size=(8, 8))(block2_output)
    
    # Flatten layer
    flatten = Flatten()(avg_pool)
    
    # Fully connected layer
    dense = Dense(units=10, activation='softmax')(flatten)

    # Construct the model
    model = Model(inputs=input_layer, outputs=dense)
    
    return model