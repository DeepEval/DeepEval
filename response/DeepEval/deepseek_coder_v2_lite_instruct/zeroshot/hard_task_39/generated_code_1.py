import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D

def dl_model():
    # Block 1
    inputs = Input(shape=(28, 28, 1))
    
    # Max Pooling Layers with different scales
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1)(inputs)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(inputs)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4)(inputs)
    
    # Flatten the results and concatenate
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    concat = Concatenate()([flatten1, flatten2, flatten3])
    
    # Fully connected layer and reshape
    fc1 = Dense(128, activation='relu')(concat)
    reshape = Reshape((1, 1, 128))(fc1)
    
    # Block 2
    branch1 = Conv2D(32, kernel_size=(1, 1), activation='relu')(reshape)
    branch2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(reshape)
    branch3 = Conv2D(32, kernel_size=(5, 5), activation='relu')(reshape)
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=1)(reshape)
    
    # Concatenate outputs from all branches
    concat_branches = Concatenate()([branch1, branch2, branch3, branch4])
    
    # Flatten the concatenated output
    flatten_concat = Flatten()(concat_branches)
    
    # Fully connected layer for classification
    fc2 = Dense(10, activation='softmax')(flatten_concat)
    
    # Create the model
    model = Model(inputs=inputs, outputs=fc2)
    
    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])