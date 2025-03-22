import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, Concatenate, Reshape

def dl_model():
    # Block 1
    input_layer = Input(shape=(28, 28, 1))
    
    # Three parallel paths with different pooling scales
    path1 = AveragePooling2D(pool_size=(1, 1), strides=1)(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_layer)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=4)(input_layer)
    
    # Flatten each pooling result
    flatten1 = Flatten()(path1)
    flatten2 = Flatten()(path2)
    flatten3 = Flatten()(path3)
    
    # Apply dropout to each flattened vector
    dropout1 = Dropout(0.25)(flatten1)
    dropout2 = Dropout(0.25)(flatten2)
    dropout3 = Dropout(0.25)(flatten3)
    
    # Concatenate the results
    concatenated = Concatenate()([dropout1, dropout2, dropout3])
    
    # Fully connected layer and reshape to 4-dimensional tensor
    fc_layer = Dense(512, activation='relu')(concatenated)
    reshaped = Reshape((1, 1, 512))(fc_layer)
    
    # Block 2
    branch1 = Conv2D(64, (1, 1), activation='relu')(reshaped)
    branch2 = Conv2D(64, (1, 1), activation='relu')(reshaped)
    branch3 = Conv2D(64, (3, 3), activation='relu')(reshaped)
    branch4 = Conv2D(64, (1, 1), activation='relu')(reshaped)
    branch5 = Conv2D(64, (3, 3), activation='relu')(reshaped)
    branch6 = Conv2D(64, (3, 3), activation='relu')(reshaped)
    branch7 = Conv2D(64, (1, 1), activation='relu')(reshaped)
    branch8 = AveragePooling2D(pool_size=(3, 3), strides=1)(reshaped)
    branch9 = Conv2D(64, (1, 1), activation='relu')(reshaped)
    
    # Concatenate the outputs of the branches
    concatenated_branches = Concatenate()([branch1, branch2, branch3, branch4, branch5, branch6, branch7, branch8, branch9])
    
    # Flatten the concatenated output
    flattened = Flatten()(concatenated_branches)
    
    # Output layer
    output_layer = Dense(10, activation='softmax')(flattened)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage:
# model = dl_model()
# model.summary()