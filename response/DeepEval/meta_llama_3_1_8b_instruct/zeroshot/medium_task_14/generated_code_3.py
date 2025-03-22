# Import necessary packages
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Dense, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model includes three sequential blocks, each comprising a convolutional layer, a batch normalization layer, 
    and a ReLU activation function to extract image features. These blocks produce three separate output paths, 
    each corresponding to one block's output. Additionally, a parallel branch of convolutional layers processes 
    the input directly. The outputs from all paths are added, and the aggregated result passes through two fully 
    connected layers for classification.
    
    Returns:
        model: The constructed deep learning model.
    """

    # Define the input layer with shape (32, 32, 3) for CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    block1_conv = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))(input_layer)
    block1_bn = BatchNormalization()(block1_conv)
    block1_relu = Activation('relu')(block1_bn)
    
    # Block 2
    block2_conv = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))(block1_relu)
    block2_bn = BatchNormalization()(block2_conv)
    block2_relu = Activation('relu')(block2_bn)
    
    # Block 3
    block3_conv = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))(block2_relu)
    block3_bn = BatchNormalization()(block3_conv)
    block3_relu = Activation('relu')(block3_bn)
    
    # Parallel branch
    parallel_conv = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))(input_layer)
    
    # Add the outputs from all paths
    merged = Add()([block3_relu, parallel_conv])
    
    # Flatten the merged output
    flat_output = Lambda(lambda x: tf.keras.layers.Flatten()(x))(merged)
    
    # Dense layers for classification
    dense1 = Dense(64, activation='relu')(flat_output)
    output_layer = Dense(10, activation='softmax')(dense1)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model