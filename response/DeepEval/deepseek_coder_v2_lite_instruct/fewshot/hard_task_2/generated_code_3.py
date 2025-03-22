import keras
from keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Process each group with a series of convolutions
    def process_group(group):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(group)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
        conv3 = Conv2D(filters=128, kernel_size=(1, 1), activation='relu')(conv2)
        return conv3
    
    processed_groups = [process_group(group) for group in split_layer]
    
    # Combine the processed groups through addition
    main_path = Add()(processed_groups)
    
    # Add the processed groups back to the original input
    combined_features = Add()([main_path, input_layer])
    
    # Flatten the combined features
    flattened_features = Flatten()(combined_features)
    
    # Feed the flattened features into a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened_features)
    
    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model