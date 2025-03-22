import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Dropout, Dense, Flatten, Reshape
from keras.models import Model

def dl_model():
    # Block 1
    input_layer = Input(shape=(28, 28, 1))
    
    # Three parallel paths with different average pooling layers
    path1 = AveragePooling2D(pool_size=(1, 1), strides=1)(input_layer)
    path1 = Flatten()(path1)
    path1 = Dropout(0.25)(path1)
    
    path2 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_layer)
    path2 = Flatten()(path2)
    path2 = Dropout(0.25)(path2)
    
    path3 = AveragePooling2D(pool_size=(4, 4), strides=4)(input_layer)
    path3 = Flatten()(path3)
    path3 = Dropout(0.25)(path3)
    
    # Concatenate the outputs of the three paths
    concat_layer = Concatenate()([path1, path2, path3])
    
    # Fully connected layer and reshape to prepare for block 2
    dense_layer = Dense(128, activation='relu')(concat_layer)
    reshape_layer = Reshape((16, 16, 1))(dense_layer)  # Since 28/1 + 28/2 + 28/4 = 16
    
    # Block 2
    branch1 = Conv2D(64, (1, 1), activation='relu')(reshape_layer)
    
    branch2 = Conv2D(64, (1, 1), activation='relu')(reshape_layer)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)
    
    branch3 = Conv2D(64, (1, 1), activation='relu')(reshape_layer)
    branch3 = Conv2D(64, (3, 3), activation='relu')(branch3)
    branch3 = Conv2D(64, (3, 3), activation='relu')(branch3)
    
    branch4 = AveragePooling2D(pool_size=(3, 3), strides=1)(reshape_layer)
    branch4 = Conv2D(64, (1, 1), activation='relu')(branch4)
    
    # Concatenate the outputs of the four branches
    final_concat = Concatenate()([branch1, branch2, branch3, branch4])
    
    # Flatten the final output for the fully connected layers
    flatten_layer = Flatten()(final_concat)
    
    # Two fully connected layers for classification
    dense1 = Dense(128, activation='relu')(flatten_layer)
    output_layer = Dense(10, activation='softmax')(dense1)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model