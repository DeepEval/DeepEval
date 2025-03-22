import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path block
    def main_path(x):
        # Convolution, dropout block
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='sigmoid')(conv1)
        drop1 = Dropout(rate=0.5)(conv1)
        
        # Convolution to restore channels
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(drop1)
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='sigmoid')(conv2)
        
        return conv2
    
    main_output = main_path(input_layer)
    
    # Branch path
    branch_input = Input(shape=(32, 32, 3))
    branch_conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(branch_input)
    branch_conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='sigmoid')(branch_conv1)
    
    # Split branch path into three groups
    group1 = Lambda(lambda tensors: keras.layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(tensors[0] if isinstance(tensors, list) else tensors))([main_output, branch_conv1, branch_conv1] * 2)
    
    # Varied kernel size paths
    def varied_kernel_paths(group_of_tensors):
        # 1x1, 3x3, 5x5 convolutions
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(group_of_tensors)
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        conv1 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(conv1)
        conv1 = MaxPooling2D(pool_size=(1, 1))(conv1)
        conv1 = Dropout(rate=0.5)(conv1)
        
        return conv1
    
    paths = varied_kernel_paths(group1)
    
    # Concatenate paths
    concat = Concatenate()(paths)
    
    # Batch normalization and flattening
    bn = BatchNormalization()(concat)
    flat = Flatten()(bn)
    
    # Dense layers
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)
    
    # Model
    model = Model(inputs=[input_layer, branch_input], outputs=output)
    
    return model