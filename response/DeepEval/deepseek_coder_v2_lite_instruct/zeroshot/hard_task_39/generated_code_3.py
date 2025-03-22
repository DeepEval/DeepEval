import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Reshape

def dl_model():
    # Block 1
    input_layer = Input(shape=(28, 28, 1))
    
    # Pooling layers with different window sizes and strides
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1)(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4)(input_layer)
    
    # Flatten each pooling result
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    
    # Concatenate the flattened results
    concat_layer = Concatenate()([flatten1, flatten2, flatten3])
    
    # Fully connected layer after Block 1
    fc_layer = Dense(128, activation='relu')(concat_layer)
    
    # Reshape the output to 4-dimensional tensor
    reshape_layer = Reshape((1, 1, 128))(fc_layer)
    
    # Block 2
    # Multiple branches for feature extraction
    branch1 = Conv2D(32, kernel_size=(1, 1), activation='relu')(reshape_layer)
    branch2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(reshape_layer)
    branch3 = Conv2D(32, kernel_size=(5, 5), activation='relu')(reshape_layer)
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=1)(reshape_layer)
    
    # Concatenate the outputs from all branches
    concat_branch_layer = Concatenate()([branch1, branch2, branch3, branch4])
    
    # Flatten the concatenated result
    flatten_layer = Flatten()(concat_branch_layer)
    
    # Fully connected layer for classification
    output_layer = Dense(10, activation='softmax')(flatten_layer)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage
model = dl_model()
model.summary()