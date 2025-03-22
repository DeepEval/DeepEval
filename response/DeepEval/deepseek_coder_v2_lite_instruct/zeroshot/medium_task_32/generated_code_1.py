import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, SeparableConv2D, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Split the input into three groups along the last dimension
    split_1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Feature extraction for each group with different kernel sizes
    conv1_1 = Conv2D(32, (1, 1), activation='relu')(split_1[0])
    conv1_3 = DepthwiseConv2D((3, 3), activation='relu')(split_1[1])
    conv1_5 = DepthwiseConv2D((5, 5), activation='relu')(split_1[2])
    
    # Concatenate the outputs of the three groups
    concatenated = Concatenate(axis=-1)([conv1_1, conv1_3, conv1_5])
    
    # Flatten the concatenated features
    flattened = Flatten()(concatenated)
    
    # Fully connected layer for classification
    outputs = Dense(10, activation='softmax')(flattened)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model