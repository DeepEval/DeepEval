import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, SeparableConv2D, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Define the input
    input_layer = Input(shape=(32, 32, 3))
    
    # First Block: Split and Depthwise Separable Convolutions
    def split_input(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)

    split_layer = Lambda(split_input)(input_layer)
    
    # Apply depthwise separable convolutions
    conv1 = SeparableConv2D(32, (1, 1), activation='relu', padding='same')(split_layer[0])
    conv2 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(split_layer[1])
    conv3 = SeparableConv2D(32, (5, 5), activation='relu', padding='same')(split_layer[2])
    
    # Concatenate the outputs
    concat1 = Concatenate()([conv1, conv2, conv3])
    
    # Second Block: Multiple Branches
    # Branch 1: 1x1 Convolution
    branch1 = Conv2D(64, (1, 1), activation='relu', padding='same')(concat1)
    
    # Branch 2: 1x1 Convolution -> 3x3 Convolution -> 3x3 Convolution
    branch2 = Conv2D(64, (1, 1), activation='relu', padding='same')(concat1)
    branch2 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)
    
    # Branch 3: 1x1 Convolution -> 3x3 Convolution
    branch3 = Conv2D(64, (1, 1), activation='relu', padding='same')(concat1)
    branch3 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch3)
    
    # Branch 4: Max Pooling -> 1x1 Convolution
    branch4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(concat1)
    branch4 = Conv2D(64, (1, 1), activation='relu', padding='same')(branch4)
    
    # Concatenate the outputs from branches
    concat2 = Concatenate()([branch1, branch2, branch3, branch4])
    
    # Flatten and Fully Connected Layer
    flatten = Flatten()(concat2)
    output = Dense(10, activation='softmax')(flatten)
    
    # Define the model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()