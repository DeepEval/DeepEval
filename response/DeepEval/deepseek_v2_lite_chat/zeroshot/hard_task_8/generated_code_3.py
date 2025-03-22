from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Concatenate, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam

def dl_model():
    # Block 1: Features Extraction
    input1 = Input(shape=(28, 28, 1))  # Assuming grayscale images (1 channel)
    x = Conv2D(32, kernel_size=(1, 1), activation='relu')(input1)
    x = DepthwiseConv2D(kernel_size=(3, 3), depth_multiplier=1, activation='relu')(x)
    x = Conv2D(64, kernel_size=(1, 1), activation='relu')(x)
    
    branch = Conv2D(64, kernel_size=(3, 3), activation='relu')(input1)
    branch = Conv2D(64, kernel_size=(1, 1), activation='relu')(branch)
    
    # Concatenate features from both paths
    merged = Concatenate(axis=-1)([x, branch])
    
    # Block 2: Feature Transformation and Classification
    shape = K.int_shape(merged)
    height = shape[1]
    width = shape[2]
    groups = shape[3]
    channels_per_group = shape[1]
    
    # Reshape to four groups with target shape
    x = Flatten()(merged)
    x = Dense(4 * 64)(x)  # Adjust the number of units based on the reshape
    x = Activation('relu')(x)
    
    # Swap dimensions
    x = Transpose()(x)
    x = Reshape((height, width, groups, channels_per_group))(x)
    
    # Channel shuffling
    x = Activation('relu')(x)
    
    # Fully connected layer for classification
    output = Dense(10, activation='softmax')(x)
    
    # Build the model
    model = Model(inputs=[input1], outputs=[output])
    
    return model

# Optional: compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])