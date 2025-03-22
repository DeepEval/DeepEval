import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, AveragePooling2D, Flatten, Dense
from keras.regularizers import l2

def dl_model():     
    # Define input layer with 3 color channels and 32x32 image size
    input_layer = Input(shape=(32, 32, 3))
    
    # Adjust input feature dimensionality to 16 using a convolutional layer
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # First level of residual connection
    def basic_block(input_tensor):
        branch = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        output = Add()([input_tensor, branch])
        return Activation('relu')(output)
    
    # Basic block 1
    block1 = basic_block(conv)
    
    # Second level of residual connection
    def residual_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
        branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        output = Add()([conv2, branch])
        return Activation('relu')(output)
    
    # Residual block 1
    block2 = residual_block(block1)
    # Residual block 2
    block3 = residual_block(block2)
    
    # Third level of residual connection
    global_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv)
    block4 = Add()([block3, global_branch])
    
    # Average pooling followed by a fully connected layer
    avg_pool = AveragePooling2D(pool_size=(8, 8))(block4)
    flatten_layer = Flatten()(avg_pool)
    dense = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=dense)
    
    return model