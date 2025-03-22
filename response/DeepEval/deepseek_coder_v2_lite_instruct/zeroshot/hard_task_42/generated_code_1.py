import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, concatenate

def dl_model():
    # Block 1
    input_layer = Input(shape=(28, 28, 1))
    
    # Path 1: Max Pooling 1x1
    path1 = MaxPooling2D(pool_size=(1, 1), strides=1)(input_layer)
    
    # Path 2: Max Pooling 2x2
    path2 = MaxPooling2D(pool_size=(2, 2), strides=2)(input_layer)
    
    # Path 3: Max Pooling 4x4
    path3 = MaxPooling2D(pool_size=(4, 4), strides=4)(input_layer)
    
    # Flatten the outputs of each path
    path1_flattened = Flatten()(path1)
    path2_flattened = Flatten()(path2)
    path3_flattened = Flatten()(path3)
    
    # Dropout regularization
    path1_flattened = Dropout(0.25)(path1_flattened)
    path2_flattened = Dropout(0.25)(path2_flattened)
    path3_flattened = Dropout(0.25)(path3_flattened)
    
    # Concatenate the flattened outputs
    concatenated = concatenate([path1_flattened, path2_flattened, path3_flattened])
    
    # Fully connected layer and reshape
    fc_layer = Dense(512, activation='relu')(concatenated)
    reshaped = tf.expand_dims(fc_layer, axis=-1)
    reshaped = tf.expand_dims(reshaped, axis=-1)
    
    # Block 2
    # Path 1: 1x1 Convolution
    path4 = Conv2D(64, (1, 1), activation='relu')(reshaped)
    
    # Path 2: 1x1 -> 1x7 -> 7x1 Convolution
    path5 = Conv2D(64, (1, 1), activation='relu')(reshaped)
    path5 = Conv2D(64, (7, 1), activation='relu')(path5)
    path5 = Conv2D(64, (1, 7), activation='relu')(path5)
    
    # Path 3: 1x1 -> 7x1 -> 1x7 Convolution
    path6 = Conv2D(64, (1, 1), activation='relu')(reshaped)
    path6 = Conv2D(64, (7, 1), activation='relu')(path6)
    path6 = Conv2D(64, (1, 7), activation='relu')(path6)
    
    # Path 4: Average Pooling with 1x1 Convolution
    path7 = Conv2D(64, (1, 1), activation='relu')(reshaped)
    path7 = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), strides=1)(path7)
    
    # Concatenate the outputs of all paths along the channel dimension
    concatenated_paths = concatenate([path4, path5, path6, path7], axis=-1)
    
    # Flatten the concatenated tensor
    flattened = Flatten()(concatenated_paths)
    
    # Fully connected layers for classification
    fc1 = Dense(256, activation='relu')(flattened)
    fc2 = Dense(10, activation='softmax')(fc1)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=fc2)
    
    return model

# Example usage
model = dl_model()
model.summary()