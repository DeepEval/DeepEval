# Import necessary packages
from tensorflow.keras.layers import Input, Concatenate, BatchNormalization, ReLU, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf

# Define the deep learning model
def dl_model():
    """
    Construct a deep learning model for image classification using the MNIST dataset.
    
    The model features two processing pathways, each employing a repeated block structure executed three times.
    The block includes batch normalization and ReLU activation, followed by a 3x3 convolutional layer that extracts features while preserving spatial dimensions.
    The original input of the block is then concatenated with the new features along the channel dimension.
    The outputs from both pathways are merged through concatenation and classified using two fully connected layers.
    
    Returns:
        model (Model): The constructed deep learning model.
    """
    
    # Define the input shape for the MNIST dataset
    input_shape = (28, 28, 1)
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # Define the first pathway
    pathway1 = inputs
    
    # Define the block for the first pathway
    def block(pathway):
        pathway = BatchNormalization()(pathway)
        pathway = ReLU()(pathway)
        pathway = Conv2D(32, (3, 3), padding='same')(pathway)
        return Concatenate()([pathway, pathway1])
    
    # Repeat the block three times for the first pathway
    for i in range(3):
        pathway1 = block(pathway1)
    
    # Define the second pathway
    pathway2 = inputs
    
    # Define the block for the second pathway
    def block2(pathway):
        pathway = BatchNormalization()(pathway)
        pathway = ReLU()(pathway)
        pathway = Conv2D(32, (3, 3), padding='same')(pathway)
        return Concatenate()([pathway, pathway2])
    
    # Repeat the block three times for the second pathway
    for i in range(3):
        pathway2 = block2(pathway2)
    
    # Merge the outputs from both pathways
    merged = Concatenate()([pathway1, pathway2])
    
    # Add a flatten layer
    merged = tf.keras.layers.Flatten()(merged)
    
    # Add a dropout layer
    merged = tf.keras.layers.Dropout(0.2)(merged)
    
    # Add a first fully connected layer
    merged = tf.keras.layers.Dense(128, activation='relu')(merged)
    
    # Add a dropout layer
    merged = tf.keras.layers.Dropout(0.2)(merged)
    
    # Add a second fully connected layer for classification
    outputs = tf.keras.layers.Dense(10, activation='softmax')(merged)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model