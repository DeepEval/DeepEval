from tensorflow.keras.layers import Input, Conv2D, Lambda, Concatenate, GlobalMaxPooling2D, Dense, Reshape, Multiply, Add, Flatten
from tensorflow.keras.models import Model
import tensorflow as tf

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Splitting input into three groups and processing each
    def block1(x):
        # Split input into three groups
        splits = tf.split(x, num_or_size_splits=3, axis=-1)
        
        # Process each group
        outputs = []
        for split in splits:
            # 1x1 convolution
            conv1 = Conv2D(32, (1, 1), activation='relu', padding='same')(split)
            # 3x3 convolution
            conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
            # Another 1x1 convolution
            conv1_again = Conv2D(32, (1, 1), activation='relu', padding='same')(conv3)
            outputs.append(conv1_again)
        
        # Concatenate the outputs from three groups
        return Concatenate()(outputs)

    block1_output = Lambda(block1)(input_layer)

    # Transition Convolution: Adjust channels to match input layer
    transition_conv = Conv2D(3, (1, 1), activation='relu', padding='same')(block1_output)

    # Block 2: Global max pooling and channel-matching weights
    pooled_output = GlobalMaxPooling2D()(transition_conv)
    
    # Fully connected layers for channel matching weights
    fc1 = Dense(64, activation='relu')(pooled_output)
    fc2 = Dense(3, activation='sigmoid')(fc1)  # Match input channels (3 channels for CIFAR-10 images)
    
    # Reshape weights to match shape of transition_conv output
    weights = Reshape((1, 1, 3))(fc2)
    
    # Multiply weights with adjusted output
    main_path_output = Multiply()([transition_conv, weights])
    
    # Add direct branch to input
    branch_output = input_layer
    added_output = Add()([main_path_output, branch_output])
    
    # Final fully connected layer for classification
    flattened = Flatten()(added_output)
    output = Dense(10, activation='softmax')(flattened)  # CIFAR-10 has 10 classes

    # Construct model
    model = Model(inputs=input_layer, outputs=output)
    return model

# Instantiate the model
model = dl_model()

# Summary of the model
model.summary()