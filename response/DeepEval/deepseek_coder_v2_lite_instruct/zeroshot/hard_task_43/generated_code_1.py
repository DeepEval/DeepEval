import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Concatenate, Reshape

def dl_model():
    # Block 1
    input_layer = Input(shape=(28, 28, 1))
    
    # Path 1: 1x1 Average Pooling
    path1 = AveragePooling2D(pool_size=(1, 1), strides=1)(input_layer)
    path1 = Flatten()(path1)
    
    # Path 2: 2x2 Average Pooling
    path2 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_layer)
    path2 = Flatten()(path2)
    
    # Path 3: 4x4 Average Pooling
    path3 = AveragePooling2D(pool_size=(4, 4), strides=4)(input_layer)
    path3 = Flatten()(path3)
    
    # Concatenate outputs of the three paths
    concat_layer = Concatenate()([path1, path2, path3])
    
    # Fully connected layer after Block 1
    fc_layer = Dense(128, activation='relu')(concat_layer)
    
    # Reshape the output of Block 1 to a 4-dimensional tensor
    reshape_layer = Reshape((1, 1, 128))(fc_layer)
    
    # Block 2
    # Branch 1: 1x1 Convolution, 3x3 Convolution
    branch1_conv1 = Conv2D(32, (1, 1), activation='relu')(reshape_layer)
    branch1_conv2 = Conv2D(64, (3, 3), activation='relu')(branch1_conv1)
    
    # Branch 2: 1x1 Convolution, 1x7 Convolution, 7x1 Convolution
    branch2_conv1 = Conv2D(32, (1, 1), activation='relu')(reshape_layer)
    branch2_conv2 = Conv2D(64, (1, 7), activation='relu')(branch2_conv1)
    branch2_conv3 = Conv2D(64, (7, 1), activation='relu')(branch2_conv2)
    
    # Branch 3: 3x3 Convolution, Average Pooling
    branch3_conv = Conv2D(64, (3, 3), activation='relu')(reshape_layer)
    branch3_avgpool = AveragePooling2D(pool_size=(3, 3), strides=1)(branch3_conv)
    
    # Concatenate outputs of the three branches
    concat_branch_layer = Concatenate()([branch1_conv2, branch2_conv3, branch3_avgpool])
    
    # Flatten the concatenated output
    flatten_layer = Flatten()(concat_branch_layer)
    
    # Fully connected layers
    fc1 = Dense(128, activation='relu')(flatten_layer)
    output_layer = Dense(10, activation='softmax')(fc1)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()