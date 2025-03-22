import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, Flatten

def block(input_tensor):
    # Global Average Pooling
    pooled_output = GlobalAveragePooling2D()(input_tensor)
    
    # Fully connected layers to produce weights
    dense1 = Dense(units=input_tensor.shape[-1], activation='relu')(pooled_output)
    dense2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense1)
    
    # Reshape to match the input shape
    reshaped_weights = Reshape((1, 1, input_tensor.shape[-1]))(dense2)
    
    # Element-wise multiplication with the input
    weighted_output = Multiply()([input_tensor, reshaped_weights])
    
    return weighted_output

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB images
    
    # Branch 1
    branch1_output = block(input_layer)
    
    # Branch 2
    branch2_output = block(input_layer)
    
    # Concatenate outputs from both branches
    concatenated_output = Concatenate()([branch1_output, branch2_output])
    
    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated_output)
    
    # Fully connected layer for final classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model