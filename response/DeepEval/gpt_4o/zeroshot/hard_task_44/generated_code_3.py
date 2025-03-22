import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, Dropout, Concatenate, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Split channels and apply convolutions with different kernel sizes
    def split_and_conv(x):
        # Split the input into 3 groups along the channel axis
        splits = tf.split(x, num_or_size_splits=3, axis=-1)
        
        # Convolution with different kernel sizes
        conv1 = Conv2D(32, (1, 1), padding='same', activation='relu')(splits[0])
        conv3 = Conv2D(32, (3, 3), padding='same', activation='relu')(splits[1])
        conv5 = Conv2D(32, (5, 5), padding='same', activation='relu')(splits[2])
        
        # Apply Dropout
        dropout1 = Dropout(0.5)(conv1)
        dropout3 = Dropout(0.5)(conv3)
        dropout5 = Dropout(0.5)(conv5)
        
        # Concatenate the outputs
        return Concatenate()([dropout1, dropout3, dropout5])
    
    block1_output = Lambda(split_and_conv)(input_layer)
    
    # Block 2: Four branches with different operations
    # Branch 1: 1x1 Convolution
    branch1 = Conv2D(64, (1, 1), padding='same', activation='relu')(block1_output)
    
    # Branch 2: <1x1 Convolution, 3x3 Convolution>
    branch2_conv1 = Conv2D(64, (1, 1), padding='same', activation='relu')(block1_output)
    branch2 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch2_conv1)
    
    # Branch 3: <1x1 Convolution, 5x5 Convolution>
    branch3_conv1 = Conv2D(64, (1, 1), padding='same', activation='relu')(block1_output)
    branch3 = Conv2D(64, (5, 5), padding='same', activation='relu')(branch3_conv1)
    
    # Branch 4: <3x3 Max Pooling, 1x1 Convolution>
    branch4_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(block1_output)
    branch4 = Conv2D(64, (1, 1), padding='same', activation='relu')(branch4_pool)
    
    # Concatenate the outputs from the four branches
    block2_output = Concatenate()([branch1, branch2, branch3, branch4])
    
    # Output layer: Flatten and fully connected
    flat = Flatten()(block2_output)
    output_layer = Dense(10, activation='softmax')(flat)  # CIFAR-10 has 10 classes
    
    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()