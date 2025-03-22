import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups
    channel_splits = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Main path: three depthwise separable conv layers with different kernel sizes
    main_path_outputs = []
    for i, split in enumerate(channel_splits):
        for size in [1, 3, 5]:
            if i == 0:  # First split
                split_conv = Conv2D(filters=64, kernel_size=(size, size), strides=(1, 1), padding='same', activation='relu')(split)
            else:
                split_conv = DepthwiseConv2D(kernel_size=(size, size), strides=(1, 1), padding='same')(split)
            split_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_conv)
            split_conv = DepthwiseConv2D(kernel_size=(size, size), strides=(1, 1), padding='same')(split_conv)
            split_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_conv)
            main_path_outputs.append(split_conv)
    
    main_path_output = Concatenate()(main_path_outputs)
    
    # Branch path: 1x1 conv layer
    branch_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path_output)
    
    # Concatenate the outputs for final classification
    concatenated_output = Concatenate()([main_path_output, branch_output])
    
    # Add fully connected layers
    dense1 = Dense(units=128, activation='relu')(concatenated_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model