import keras
from keras.layers import Input, Lambda, DepthwiseConv2D, BatchNormalization, Concatenate, Conv2D, AveragePooling2D, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First block: Split input and apply depthwise separable convolutions
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Depthwise separable convolutions with different kernel sizes
    path1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[0])
    path2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_inputs[1])
    path3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_inputs[2])
    
    # Batch normalization on the outputs
    path1 = BatchNormalization()(path1)
    path2 = BatchNormalization()(path2)
    path3 = BatchNormalization()(path3)
    
    # Concatenate the outputs
    block1_output = Concatenate()([path1, path2, path3])
    
    # Second block: Multiple branches for feature extraction
    branch1 = Concatenate()([
        Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(block1_output),
        Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(block1_output)
    ])
    
    branch2 = Concatenate()([
        Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(block1_output),
        Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(block1_output),
        Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(block1_output),
        Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(block1_output)
    ])
    
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(block1_output)
    
    # Concatenate all branches
    block2_output = Concatenate()([branch1, branch2, branch3])
    
    # Flatten and fully connected layers
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model