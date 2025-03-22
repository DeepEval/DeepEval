import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three channel groups
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Define the feature extraction paths
    def feature_extraction_path(input_tensor, kernel_size):
        x = Conv2D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(input_tensor)
        x = BatchNormalization()(x)
        return x
    
    # Apply feature extraction to each channel group
    path1 = feature_extraction_path(split_layer[0], (1, 1))
    path2 = feature_extraction_path(split_layer[1], (3, 3))
    path3 = feature_extraction_path(split_layer[2], (5, 5))
    
    # Concatenate the outputs of the three paths
    concatenated_output = Concatenate()([path1, path2, path3])
    
    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated_output)
    
    # Pass through fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model