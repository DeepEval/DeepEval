import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Permute

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Main path of the model
    def block1(x):
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        return x
    
    def block2(x):
        shape = keras.backend.int_shape(x)
        x = Lambda(lambda tensors: keras.backend.reshape(tensors, (-1, shape[1], shape[2], 3)))([x, shape])
        x = Permute((3, 1, 2))(x)  # Swap dimensions
        x = keras.backend.reshape(x, (-1, shape[1], shape[2], 3))
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        return x
    
    def block3(x):
        x = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='valid', activation='relu')(x)
        x = DepthwiseConv2D((3, 1), strides=(1, 1), padding='valid')(x)  # Depthwise separable convolution
        return x
    
    block1_output = block1(inputs)
    block2_output = block2(block1_output)
    block3_output = block3(block2_output)
    
    # Branch path
    branch_output = Lambda(lambda tensors: keras.backend.average(tensors))(inputs)
    
    # Concatenate outputs from both paths
    concatenated = Concatenate()([block3_output, branch_output])
    
    # Batch normalization, flattening, dense layers
    bn = BatchNormalization()(concatenated)
    flat = Flatten()(bn)
    
    # Fully connected layer
    dense = Dense(units=128, activation='relu')(flat)
    output = Dense(units=10, activation='softmax')(dense)
    
    # Model construction
    model = keras.Model(inputs=inputs, outputs=output)
    
    return model

# Create the model
model = dl_model()
model.summary()