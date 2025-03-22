import keras
from keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Add
import tensorflow as tf

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    def split_channels(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)
    
    split_layer = Lambda(split_channels)(input_layer)
    
    processed_splits = []
    
    # Process each split through a series of convolutions
    for split in split_layer:
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(split)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(conv2)
        processed_splits.append(conv3)
    
    # Combine the outputs from the three groups using addition
    main_path = Add()(processed_splits)
    
    # Fuse the main path with the original input layer
    fused_path = Add()([main_path, input_layer])
    
    # Flatten the result and add a fully connected layer for classification
    flatten_layer = Flatten()(fused_path)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model