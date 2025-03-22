import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate, Reshape, Add, BatchNormalization

def dl_model():
    # Block 1
    input_layer = Input(shape=(28, 28, 1))
    
    # Path 1: 1x1 Max Pooling
    path1 = MaxPooling2D(pool_size=(1, 1), strides=1)(input_layer)
    
    # Path 2: 2x2 Max Pooling
    path2 = MaxPooling2D(pool_size=(2, 2), strides=2)(input_layer)
    
    # Path 3: 4x4 Max Pooling
    path3 = MaxPooling2D(pool_size=(4, 4), strides=4)(input_layer)
    
    # Concatenate the outputs of the three paths
    concat1 = Concatenate(axis=-1)([Flatten()(path1), Flatten()(path2), Flatten()(path3)])
    
    # Dropout
    dropout1 = Dropout(0.5)(concat1)
    
    # Fully connected layer and reshape
    dense1 = Dense(128, activation='relu')(dropout1)
    reshape1 = Reshape((1, 1, 128))(dense1)
    
    # Block 2
    # Path 4: 1x1 Convolution
    path4_1 = Conv2D(64, kernel_size=(1, 1), activation='relu')(reshape1)
    
    # Path 5: 1x1 followed by 1x7 Convolution
    path5_1 = Conv2D(64, kernel_size=(1, 1), activation='relu')(reshape1)
    path5_2 = Conv2D(64, kernel_size=(1, 7), activation='relu')(path5_1)
    
    # Path 6: 1x1 followed by 7x1 Convolution
    path6_1 = Conv2D(64, kernel_size=(1, 1), activation='relu')(reshape1)
    path6_2 = Conv2D(64, kernel_size=(7, 1), activation='relu')(path6_1)
    
    # Path 7: 1x1 Convolution followed by average pooling
    path7_1 = Conv2D(64, kernel_size=(1, 1), activation='relu')(reshape1)
    path7_2 = AveragePooling2D(pool_size=(7, 7))(path7_1)
    
    # Concatenate the outputs of the four paths
    concat2 = Concatenate(axis=-1)([path4_1, path5_2, path6_2, path7_2])
    
    # Flatten the concatenated output
    flatten2 = Flatten()(concat2)
    
    # Fully connected layer
    dense2 = Dense(128, activation='relu')(flatten2)
    
    # Output layer
    output_layer = Dense(10, activation='softmax')(dense2)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model