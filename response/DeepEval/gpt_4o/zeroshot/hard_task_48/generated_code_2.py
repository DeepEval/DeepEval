import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, SeparableConv2D, BatchNormalization, Concatenate, Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Split and Separable Convolutions
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # 1x1 Separable Convolution
    conv1x1 = SeparableConv2D(32, (1, 1), padding='same', activation='relu')(split_layer[0])
    bn1x1 = BatchNormalization()(conv1x1)
    
    # 3x3 Separable Convolution
    conv3x3 = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(split_layer[1])
    bn3x3 = BatchNormalization()(conv3x3)
    
    # 5x5 Separable Convolution
    conv5x5 = SeparableConv2D(32, (5, 5), padding='same', activation='relu')(split_layer[2])
    bn5x5 = BatchNormalization()(conv5x5)
    
    # Concatenate outputs of Block 1
    block1_output = Concatenate()([bn1x1, bn3x3, bn5x5])
    
    # Block 2: Parallel Paths
    # Path 1: 1x1 Conv
    path1 = Conv2D(64, (1, 1), activation='relu', padding='same')(block1_output)
    
    # Path 2: AvgPooling -> 1x1 Conv
    path2_avgpool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(block1_output)
    path2 = Conv2D(64, (1, 1), activation='relu', padding='same')(path2_avgpool)
    
    # Path 3: 1x1 Conv -> (1x3, 3x1)
    path3_conv1 = Conv2D(64, (1, 1), activation='relu', padding='same')(block1_output)
    path3_conv2a = Conv2D(64, (1, 3), activation='relu', padding='same')(path3_conv1)
    path3_conv2b = Conv2D(64, (3, 1), activation='relu', padding='same')(path3_conv1)
    path3 = Concatenate()([path3_conv2a, path3_conv2b])
    
    # Path 4: 1x1 Conv -> 3x3 Conv -> (1x3, 3x1)
    path4_conv1 = Conv2D(64, (1, 1), activation='relu', padding='same')(block1_output)
    path4_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(path4_conv1)
    path4_conv3a = Conv2D(64, (1, 3), activation='relu', padding='same')(path4_conv2)
    path4_conv3b = Conv2D(64, (3, 1), activation='relu', padding='same')(path4_conv2)
    path4 = Concatenate()([path4_conv3a, path4_conv3b])
    
    # Concatenate outputs of Block 2
    block2_output = Concatenate()([path1, path2, path3, path4])
    
    # Flatten and Fully Connected Layer
    flatten = Flatten()(block2_output)
    output_layer = Dense(10, activation='softmax')(flatten)
    
    # Build Model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model