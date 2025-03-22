import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, Add, Flatten, Dense, Concatenate

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # Main Path
    # Split the input into three groups along the channel axis
    split_1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Multi-scale feature extraction with separable convolutional layers
    conv_3x3_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(split_1[0])
    conv_5x5_1 = Conv2D(32, (5, 5), padding='same', activation='relu')(split_1[1])
    conv_1x1_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(split_1[2])
    
    # Concatenate the outputs from the three groups
    main_path_output = Concatenate(axis=-1)([conv_3x3_1, conv_5x5_1, conv_1x1_1])
    
    # Branch Path
    # Apply a 1x1 convolutional layer to the input to align the number of output channels
    branch_path_output = Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    
    # Fuse both paths through addition
    fused_output = Add()([main_path_output, branch_path_output])
    
    # Flatten the output and pass through two fully connected layers
    flattened_output = Flatten()(fused_output)
    dense_1 = Dense(128, activation='relu')(flattened_output)
    dense_2 = Dense(10, activation='softmax')(dense_1)  # 10-class classification
    
    # Define the model
    model = Model(inputs=inputs, outputs=dense_2)
    
    return model

# Create and print the model summary
model = dl_model()
model.summary()