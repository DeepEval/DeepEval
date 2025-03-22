import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Multiply
from tensorflow.keras.layers import Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal

def dl_model():
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Create the input layer
    inputs = Input(shape=input_shape, name='input')
    
    # Block 1: Process the input through four parallel branches
    branch1 = Conv2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(inputs)
    branch1 = BatchNormalization()(branch1)
    branch1 = LeakyReLU(alpha=0.2)(branch1)
    
    branch2 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    branch2 = BatchNormalization()(branch2)
    branch2 = LeakyReLU(alpha=0.2)(branch2)
    
    branch3 = Conv2D(32, (5, 5), padding='same', kernel_initializer='he_normal')(inputs)
    branch3 = BatchNormalization()(branch3)
    branch3 = LeakyReLU(alpha=0.2)(branch3)
    
    branch4 = MaxPooling2D((3, 3), strides=2, padding='same')(inputs)
    
    # Concatenate the outputs of the four branches
    x = Concatenate()([branch1, branch2, branch3, branch4])
    
    # Block 2: Reduce the dimensionality of the feature maps using global average pooling
    x = GlobalAveragePooling2D()(x)
    
    # Define the number of units in the fully connected layers
    num_units = x.shape[-1]
    
    # Fully connected layers
    x = Dense(num_units, activation='relu', kernel_initializer='he_normal')(x)
    x = Dense(num_units, activation='relu', kernel_initializer='he_normal')(x)
    
    # Reshape the weights to match the input's shape
    weights = Reshape((32, 32, 1))(x)
    
    # Element-wise multiplication of the weights with the input feature map
    x = Multiply()([inputs, weights])
    
    # Final fully connected layer
    outputs = Dense(10, activation='softmax', kernel_initializer='he_normal')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Compile the model with Adam optimizer and categorical cross-entropy loss
model = dl_model()
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
print(model.summary())