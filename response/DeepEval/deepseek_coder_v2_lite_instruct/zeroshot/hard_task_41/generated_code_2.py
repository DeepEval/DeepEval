import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, AveragePooling2D, Concatenate

def dl_model():
    # Block 1
    input_layer = Input(shape=(28, 28, 1))
    
    # Three parallel paths
    path1 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    path1 = AveragePooling2D(pool_size=(1, 1), strides=1)(path1)
    path1 = Flatten()(path1)
    path1 = Dropout(0.5)(path1)
    
    path2 = Conv2D(64, (2, 2), activation='relu')(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=2)(path2)
    path2 = Flatten()(path2)
    path2 = Dropout(0.5)(path2)
    
    path3 = Conv2D(64, (4, 4), activation='relu')(input_layer)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=4)(path3)
    path3 = Flatten()(path3)
    path3 = Dropout(0.5)(path3)
    
    # Concatenate outputs of the three paths
    concat_layer = Concatenate()([path1, path2, path3])
    
    # Fully connected layer and reshape
    fc_layer = Dense(128, activation='relu')(concat_layer)
    reshape_layer = tf.reshape(fc_layer, (-1, 1, 1, 128))
    
    # Block 2
    branch1 = Conv2D(64, (1, 1), activation='relu')(reshape_layer)
    
    branch2 = Conv2D(64, (1, 1), activation='relu')(reshape_layer)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)
    
    branch3 = Conv2D(64, (1, 1), activation='relu')(reshape_layer)
    branch3 = Conv2D(64, (3, 3), activation='relu')(branch3)
    branch3 = Conv2D(64, (3, 3), activation='relu')(branch3)
    
    branch4 = AveragePooling2D(pool_size=(3, 3), strides=1)(reshape_layer)
    branch4 = Conv2D(64, (1, 1), activation='relu')(branch4)
    
    # Concatenate outputs of the four branches
    concat_branch_layer = Concatenate()([branch1, branch2, branch3, branch4])
    
    # Flatten the output for the final classification
    flatten_layer = Flatten()(concat_branch_layer)
    
    # Two fully connected layers for classification
    output_layer = Dense(10, activation='softmax')(flatten_layer)
    
    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage
model = dl_model()
model.summary()