import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate, AveragePooling2D

def dl_model():
    # Block 1
    inputs = Input(shape=(28, 28, 1))
    
    # Three parallel paths with different pooling scales
    path1 = AveragePooling2D(pool_size=(1, 1), strides=1)(inputs)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=2)(inputs)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=4)(inputs)
    
    # Flatten each path
    path1 = Flatten()(path1)
    path2 = Flatten()(path2)
    path3 = Flatten()(path3)
    
    # Apply dropout to each path
    path1 = Dropout(0.5)(path1)
    path2 = Dropout(0.5)(path2)
    path3 = Dropout(0.5)(path3)
    
    # Concatenate the outputs of the three paths
    concatenated = Concatenate()([path1, path2, path3])
    
    # Fully connected layer and reshape
    fc_layer = Dense(128, activation='relu')(concatenated)
    reshaped = tf.expand_dims(fc_layer, axis=-1)
    reshaped = tf.expand_dims(reshaped, axis=-1)
    
    # Block 2
    # Four branches for feature extraction
    branch1 = Conv2D(32, (1, 1), activation='relu')(reshaped)
    branch2 = Conv2D(32, (1, 1), activation='relu')(reshaped)
    branch3 = Conv2D(32, (3, 3), activation='relu')(reshaped)
    branch4 = Conv2D(32, (1, 1), activation='relu')(reshaped)
    branch5 = Conv2D(32, (3, 3), activation='relu')(reshaped)
    branch6 = Conv2D(32, (3, 3), activation='relu')(reshaped)
    branch7 = AveragePooling2D(pool_size=(3, 3), strides=1)(reshaped)
    branch8 = Conv2D(32, (1, 1), activation='relu')(branch7)
    
    # Concatenate the outputs of the branches
    concatenated_branches = Concatenate()([branch1, branch2, branch3, branch4, branch5, branch6, branch8])
    
    # Flatten the concatenated output
    flattened = Flatten()(concatenated_branches)
    
    # Output layer
    outputs = Dense(10, activation='softmax')(flattened)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage
model = dl_model()
model.summary()