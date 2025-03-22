import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense

def dl_model():
    # Block 1: Main path and branch path
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_bn = BatchNormalization()(conv1)
    conv1_bn_act = keras.activations.relu(conv1_bn)
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_bn_act)
    conv2_bn = BatchNormalization()(conv2)
    conv2_bn_act = keras.activations.relu(conv2_bn)
    
    # Branch path
    branch_input = input_layer
    branch_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_input)
    branch_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_input)
    
    # Concatenate outputs from both paths
    concat_layer = Add()([conv2_bn_act, branch_conv2])
    
    # Block 2: Three max pooling layers
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(concat_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(concat_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(concat_layer)
    
    # Flatten and concatenate pooling layers
    flatten_pool = Flatten()(pool3)
    concat_pool = Concatenate()([flatten_pool, pool2, pool1])
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concat_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model