from keras.models import Model
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Flatten, Dense
from keras.layers import BatchNormalization, Activation
from keras.layers import Add

def dl_model():
    # Input shape for the model
    input_shape = (28, 28, 1)
    
    # First branch
    branch1 = Conv2D(32, (1, 1), padding='same')(input_shape)
    branch1 = Activation('relu')(branch1)
    branch1 = DepthwiseConv2D((3, 3), depth_multiplier=1, strides=(1, 1), padding='same')(branch1)
    branch1 = Activation('relu')(branch1)
    branch1 = Conv2D(64, (1, 1), padding='same')(branch1)
    
    # Second branch
    branch2 = Conv2D(32, (1, 1), padding='same')(input_shape)
    branch2 = Activation('relu')(branch2)
    branch2 = DepthwiseConv2D((3, 3), depth_multiplier=1, strides=(1, 1), padding='same')(branch2)
    branch2 = Activation('relu')(branch2)
    branch2 = Conv2D(64, (1, 1), padding='same')(branch2)
    
    # Third branch
    branch3 = Conv2D(32, (1, 1), padding='same')(input_shape)
    branch3 = Activation('relu')(branch3)
    branch3 = DepthwiseConv2D((3, 3), depth_multiplier=1, strides=(1, 1), padding='same')(branch3)
    branch3 = Activation('relu')(branch3)
    branch3 = Conv2D(64, (1, 1), padding='same')(branch3)
    
    # Concatenate outputs from the branches
    concat = Concatenate(axis=-1)([branch1, branch2, branch3])
    
    # Flatten and fully connected layers for classification
    flatten = Flatten()(concat)
    output = Dense(10, activation='softmax')(flatten)
    
    # Create the model
    model = Model(inputs=input_shape, outputs=output)
    
    return model

# Example usage:
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])