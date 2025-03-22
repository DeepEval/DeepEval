import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First block for feature extraction
    split1 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    conv1 = split1[0]  # 1x1 conv
    conv2 = split1[1]  # 3x3 conv
    conv3 = split1[2]  # 5x5 conv
    
    # Separable conv layers
    conv1_separable = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(conv1)
    conv2_separable = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(conv2)
    conv3_separable = Conv2D(filters=32, kernel_size=(5, 5), padding='same')(conv3)
    
    # Concatenate the results
    concat1 = Concatenate(axis=-1)([conv1_separable, conv2_separable, conv3_separable])
    
    # Second block for feature extraction
    split2 = Lambda(lambda x: keras.backend.split(x, 2, axis=-1))(concat1)
    conv1_branch = split2[0]  # 1x1 conv
    conv2_branch = split2[1]  # 3x3 conv
    
    # Max pooling branch
    maxpool = MaxPooling2D(pool_size=(1, 1))(input_layer)
    conv_branch = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(maxpool)
    
    # Convolution branches
    conv3_branch1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(conv1_branch)
    conv3_branch2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(conv2_branch)
    
    # Concatenate the feature branches
    concat2 = Concatenate(axis=-1)([conv1_branch, conv2_branch, conv3_branch1, conv3_branch2])
    
    # Flatten and dense layers
    flatten = Flatten()(concat2)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model