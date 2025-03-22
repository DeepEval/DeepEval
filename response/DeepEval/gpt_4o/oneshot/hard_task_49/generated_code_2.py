import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dense, Reshape, Lambda, SeparableConv2D, Concatenate
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First Block: Three Average Pooling Layers
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    
    # Flatten each pooling result
    flat1 = Flatten()(avg_pool1)
    flat2 = Flatten()(avg_pool2)
    flat3 = Flatten()(avg_pool3)
    
    # Concatenate flattened vectors
    concat = Concatenate()([flat1, flat2, flat3])
    
    # Fully connected layer
    fc1 = Dense(units=256, activation='relu')(concat)
    
    # Reshape to 4D tensor
    reshape = Reshape((4, 4, 16))(fc1)  # Adjust dimensions to ensure it fits MNIST input requirements
    
    # Second Block: Split input into four groups
    def split_and_process(input_tensor):
        split_tensors = tf.split(input_tensor, num_or_size_splits=4, axis=-1)
        processed_tensors = []
        kernel_sizes = [(1, 1), (3, 3), (5, 5), (7, 7)]
        
        for i, kernel_size in enumerate(kernel_sizes):
            processed = SeparableConv2D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(split_tensors[i])
            processed_tensors.append(processed)
        
        return Concatenate()(processed_tensors)
    
    processed_output = Lambda(split_and_process)(reshape)
    
    # Flatten and final fully connected layer
    flatten_layer = Flatten()(processed_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model