import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, SeparableConv2D, Conv2D, Add, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    # Split the input along the channel dimension into 3 groups
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Apply depthwise separable convolutions with different kernel sizes
    conv1x1 = SeparableConv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv3x3 = SeparableConv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(split_layer[1])
    conv5x5 = SeparableConv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu')(split_layer[2])
    
    # Concatenate the outputs
    main_path_output = Concatenate()([conv1x1, conv3x3, conv5x5])
    
    # Branch path
    branch_output = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine main path and branch path
    combined_output = Add()([main_path_output, branch_output])
    
    # Flatten and apply fully connected layers
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model