import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First block for feature extraction
    split_layer = Lambda(lambda x: keras.layers.split(x, 3, axis=-1))(input_layer)
    
    # Path 1: depthwise separable 1x1 convolution
    path1 = split_layer[0]
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', use_depthwise=True)(path1)
    batch_norm1 = BatchNormalization()(conv1)
    
    # Path 2: depthwise separable 3x3 convolution
    path2 = split_layer[1]
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', use_depthwise=True)(path2)
    batch_norm2 = BatchNormalization()(conv2)
    
    # Path 3: depthwise separable 5x5 convolution
    path3 = split_layer[2]
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', use_depthwise=True)(path3)
    batch_norm3 = BatchNormalization()(conv3)
    
    # Concatenate outputs from the three paths
    concatenated = Concatenate()(
        [conv1, conv2, conv3]
    )
    
    # Second block for feature extraction
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concatenated)
    branch3 = Conv2D(filters=64, kernel_size=(1, 7), (1, 7), padding='valid', activation='relu')(concatenated)
    branch4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concatenated)
    
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(branch1)
    pool2 = MaxPooling2D(pool_size=(1, 7), strides=(1, 1), padding='valid')(branch2)
    pool3 = MaxPooling2D(pool_size=(7, 1), strides=(1, 1), padding='same')(branch3)
    pool4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(branch4)
    
    concatenated_branch = Concatenate()(
        [pool1, pool2, pool3, pool4]
    )
    
    flatten = Flatten()(concatenated_branch)
    
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model