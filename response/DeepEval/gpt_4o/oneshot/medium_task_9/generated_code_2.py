import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from keras.models import Model

def basic_block(input_tensor, filters, kernel_size=(3, 3), padding='same', strides=(1, 1)):
    # Main path with Conv -> BatchNorm -> ReLU
    conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)(input_tensor)
    bn = BatchNormalization()(conv)
    main_path_output = ReLU()(bn)
    
    # Branch path
    branch_path_output = Conv2D(filters=filters, kernel_size=(1, 1), padding='same', strides=strides)(input_tensor)
    
    # Feature fusion using addition
    output = Add()([main_path_output, branch_path_output])
    
    return output

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial dimensionality reduction
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Two consecutive basic blocks
    block1_output = basic_block(initial_conv, filters=16)
    block2_output = basic_block(block1_output, filters=16)
    
    # Average pooling layer
    avg_pool = AveragePooling2D(pool_size=(8, 8))(block2_output)
    
    # Flatten the feature map
    flatten_layer = Flatten()(avg_pool)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model