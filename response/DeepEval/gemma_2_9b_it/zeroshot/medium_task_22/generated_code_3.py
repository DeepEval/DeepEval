import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate

def dl_model():
    
    # Input Layer
    inputs = Input(shape=(32, 32, 3)) 

    # Branch 1: 3x3 convolutions
    x_branch1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    x_branch1 = Conv2D(32, (3, 3), activation='relu')(x_branch1)

    # Branch 2: 1x1 -> 3x3 -> 3x3 convolutions
    x_branch2 = Conv2D(64, (1, 1), activation='relu')(inputs)
    x_branch2 = Conv2D(32, (3, 3), activation='relu')(x_branch2)
    x_branch2 = Conv2D(32, (3, 3), activation='relu')(x_branch2)

    # Branch 3: Max Pooling
    x_branch3 = MaxPooling2D((2, 2))(inputs)
    x_branch3 = Conv2D(32, (3, 3), activation='relu')(x_branch3)

    # Concatenate Branches
    x = concatenate([x_branch1, x_branch2, x_branch3])

    # Flatten and Fully Connected Layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)  

    # Create Model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

# Example Usage
model = dl_model()
model.summary()