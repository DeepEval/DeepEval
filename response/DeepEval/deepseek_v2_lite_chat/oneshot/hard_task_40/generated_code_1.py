import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, BatchNormalization, Reshape
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Three average pooling layers
    conv_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool_1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_1)
    pool_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(pool_1)
    pool_3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(pool_2)
    
    # Flatten and concatenate
    flat_1 = Flatten()(pool_3)
    concat = Concatenate()([flat_1])
    
    # Dense layer followed by reshape
    dense_1 = Dense(units=128, activation='relu')(concat)
    reshape = Reshape((-1, 1))(dense_1)
    
    # Second Block
    def block(input_tensor):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Path 2: 3x3 convolution
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
        # Path 3: 5x5 convolution
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(path2)
        # Path 4: 1x1 max pooling
        path4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(path3)
        # Dropout layer for regularization
        path4 = Dropout(0.5)(path4)
        # Concatenate
        concat_path = Concatenate()([path1, path2, path3, path4])
        return concat_path
    
    block_output = block(reshape)
    # Batch normalization
    bn = BatchNormalization()(block_output)
    # Dense layer
    dense_2 = Dense(units=128, activation='relu')(bn)
    # Second output layer
    output_layer = Dense(units=10, activation='softmax')(dense_2)
    
    # Model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()