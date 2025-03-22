import tensorflow as tf
from keras.layers import Input, Conv2D, GlobalMaxPooling2D, Dense, Add, Multiply, Reshape, Lambda, Concatenate
from keras.models import Model

def dl_model():
    # CIFAR-10 images are 32x32 RGB images
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    # Split input into three parts along the last axis
    def split_and_process(input_tensor):
        # Use tf.split to split the tensor into 3 along the channel axis
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        processed_splits = []
        for split in splits:
            # 1x1 convolution
            conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(split)
            # 3x3 convolution
            conv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
            # 1x1 convolution
            conv1_again = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(conv3)
            processed_splits.append(conv1_again)
        
        # Concatenate processed splits
        output = Concatenate()(processed_splits)
        return output
    
    block1_output = split_and_process(input_layer)

    # Transition Convolution Layer - adjust channels to match input layer
    transition_conv = Conv2D(filters=3, kernel_size=(1, 1), activation='relu', padding='same')(block1_output)

    # Block 2
    # Global max pooling
    global_max_pool = GlobalMaxPooling2D()(transition_conv)

    # Fully connected layers to generate channel-matching weights
    fc1 = Dense(units=32, activation='relu')(global_max_pool)
    fc2 = Dense(units=3, activation='sigmoid')(fc1)  # Assuming we want channel-wise weights

    # Reshape to match the channels of transition_conv
    reshape_weights = Reshape((1, 1, 3))(fc2)

    # Multiply weights with the adjusted output
    scaled_transition_conv = Multiply()([transition_conv, reshape_weights])

    # Add a branch directly from the input
    main_branch_output = Add()([scaled_transition_conv, input_layer])

    # Fully connected layer for classification
    final_output = Dense(units=10, activation='softmax')(main_branch_output)

    # Create the model
    model = Model(inputs=input_layer, outputs=final_output)

    return model