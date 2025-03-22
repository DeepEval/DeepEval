import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First block: Feature extraction using separable convolutions with different kernel sizes
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Separate convolutional layers for each group
    conv1x1 = Conv2D(32, (1, 1), activation='relu')(split_layer[0])
    conv3x3 = Conv2D(32, (3, 3), padding='same', activation='relu')(split_layer[1])
    conv5x5 = Conv2D(32, (5, 5), padding='same', activation='relu')(split_layer[2])
    
    # Concatenate the outputs
    concat_layer1 = Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5])
    
    # Second block: Enhanced feature extraction with multiple branches
    # Branch 1: 3x3 convolution
    branch1 = Conv2D(64, (3, 3), activation='relu')(concat_layer1)
    
    # Branch 2: 1x1 -> 3x3 -> 3x3 convolutions
    branch2a = Conv2D(64, (1, 1), activation='relu')(concat_layer1)
    branch2b = Conv2D(64, (3, 3), padding='same', activation='relu')(branch2a)
    branch2c = Conv2D(64, (3, 3), padding='same', activation='relu')(branch2b)
    
    # Branch 3: Max pooling
    branch3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(concat_layer1)
    
    # Concatenate outputs from all branches
    concat_layer2 = Concatenate(axis=-1)([branch1, branch2c, branch3])
    
    # Global average pooling and fully connected layer
    gap_layer = GlobalAveragePooling2D()(concat_layer2)
    output_layer = Dense(10, activation='softmax')(gap_layer)
    
    # Create and return the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Example usage
model = dl_model()
model.summary()