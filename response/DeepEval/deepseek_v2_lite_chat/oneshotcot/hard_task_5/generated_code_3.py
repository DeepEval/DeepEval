import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split input into three groups
    group_1, group_2, group_3 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    
    # Process each group through a 1x1 convolutional layer
    conv_1 = Conv2D(filters=int(3/3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group_1)
    conv_2 = Conv2D(filters=int(3/3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group_2)
    conv_3 = Conv2D(filters=int(3/3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group_3)
    
    # Concatenate along the channel dimension
    fused_features = Concatenate(axis=-1)([conv_1, conv_2, conv_3])

    # Block 1
    block_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(fused_features)
    block_1 = MaxPooling2D(pool_size=(2, 2))(block_1)
    
    # Block 2
    shape_block_1 = keras.backend.int_shape(block_1)
    reshaped_block_1 = keras.backend.reshape(block_1, (shape_block_1[0]*shape_block_1[1], shape_block_1[2], shape_block_1[3]))
    reshaped_block_1 = keras.backend.permute_dimensions(reshaped_block_1, (0, 2, 3, 1))
    reshaped_block_1 = keras.backend.reshape(reshaped_block_1, (shape_block_1[0], shape_block_1[1]*shape_block_1[2], shape_block_1[3]))
    
    block_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_block_1)
    block_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block_2)
    
    # Block 3
    block_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block_2)
    block_3 = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block_3)
    
    # Branch
    branch_output = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine main path and branch outputs
    combined_output = keras.backend.concatenate([block_3, branch_output], axis=-1)
    
    # Fully connected layer
    dense = Dense(units=128, activation='relu')(combined_output)
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model