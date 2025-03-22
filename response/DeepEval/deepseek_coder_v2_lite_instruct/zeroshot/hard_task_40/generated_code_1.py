import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D, Dropout

def dl_model():
    # Define the input layer
    inputs = Input(shape=(28, 28, 1))

    # First block
    x1 = AveragePooling2D(pool_size=(1, 1), strides=1)(inputs)
    x2 = AveragePooling2D(pool_size=(2, 2), strides=2)(inputs)
    x3 = AveragePooling2D(pool_size=(4, 4), strides=4)(inputs)
    
    # Flatten and concatenate the outputs from the pooling layers
    x_concat = Concatenate()([Flatten()(x1), Flatten()(x2), Flatten()(x3)])
    
    # Reshape the concatenated output into a 4-dimensional tensor
    x_reshaped = Reshape((4,))(x_concat)
    
    # Second block
    # Path 1: 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    path1 = Dropout(0.5)(path1)
    
    # Path 2: 1x1 followed by two 3x3 convolutions
    path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(path2)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(path2)
    path2 = Dropout(0.5)(path2)
    
    # Path 3: 1x1 followed by a single 3x3 convolution
    path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    path3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(path3)
    path3 = Dropout(0.5)(path3)
    
    # Path 4: 1x1 convolution followed by average pooling
    path4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    path4 = AveragePooling2D(pool_size=(2, 2), strides=2)(path4)
    path4 = Dropout(0.5)(path4)
    
    # Concatenate the outputs from all paths along the channel dimension
    x_concat_paths = Concatenate(axis=-1)([path1, path2, path3, path4])
    
    # Fully connected layers
    x_fc1 = Dense(128, activation='relu')(x_concat)
    x_fc2 = Dense(10, activation='softmax')(x_fc1)
    
    # Define the model
    model = Model(inputs=inputs, outputs=x_fc2)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model