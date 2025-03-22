import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Permute

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Reshape the input to three groups
    reshaped_layer = Reshape((3, 32, 32))(input_layer)
    
    # Swap the third and fourth dimensions
    permuted_layer = Permute((2, 0, 1), name='permute_axes')(reshaped_layer)
    
    # Split the groups into three tensors
    tensors = keras.layers.Lambda(lambda tensors: keras.layers.stack(tensors, name='split_tensors'))([permuted_layer, reshaped_layer, reshaped_layer])
    
    # Reshape the tensors back to the original input shape
    reshaped_tensors = Reshape((32, 32, 3), name='reshape_to_input_shape')(tensors)
    
    # Flatten each tensor and concatenate
    flat_concat = keras.layers.Concatenate()(tensors)
    
    # Fully connected layer
    dense_layer = Dense(units=1000, activation='relu')(flat_concat)
    
    # Output layer with softmax activation
    output_layer = Dense(units=10, activation='softmax')(dense_layer)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model