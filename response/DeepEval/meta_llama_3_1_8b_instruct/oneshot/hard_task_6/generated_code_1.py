import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Reshape, Permute, DepthwiseConv2D, AveragePooling2D
from keras.backend import tf as ktf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1_output = Lambda(lambda x: ktf.split(x, num_or_size_splits=3, axis=3))(conv1)
    block1_output = [Conv2D(filters=int(64/3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x) for x in block1_output]
    block1_output = Concatenate()(block1_output)
    
    # Block 2
    block2_input_shape = ktf.shape(block1_output)
    block2_output = Reshape((block2_input_shape[1], block2_input_shape[2], 3, int(64/3)))(block1_output)
    block2_output = Permute((2, 3, 1, 4))(block2_output)
    block2_output = Reshape((block2_input_shape[1], block2_input_shape[2], int(64/3)*3))(block2_output)
    
    # Block 3
    block3_output = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_output)
    
    # Repeat Block 1
    repeated_block1_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block3_output)
    repeated_block1_output = Lambda(lambda x: ktf.split(x, num_or_size_splits=3, axis=3))(repeated_block1_output)
    repeated_block1_output = [Conv2D(filters=int(64/3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x) for x in repeated_block1_output]
    repeated_block1_output = Concatenate()(repeated_block1_output)
    
    # Branch Path
    branch_output = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    branch_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_output)
    
    # Concatenate Outputs
    output = Concatenate()([block3_output, repeated_block1_output, branch_output])
    
    # Classification
    output = BatchNormalization()(output)
    output = Flatten()(output)
    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=64, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)
    
    model = keras.Model(inputs=input_layer, outputs=output)

    return model