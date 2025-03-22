import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, Flatten
from keras.models import Model

def block(input_tensor):
    # Apply global average pooling
    pooled_output = GlobalAveragePooling2D()(input_tensor)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(pooled_output)
    dense2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense1)  # Output dimension matches input channels
    
    # Reshape the output to match the input shape
    reshape_output = Reshape((1, 1, input_tensor.shape[-1]))(dense2)  # Reshape to 1x1xChannels

    # Element-wise multiplication with the original input
    scaled_output = Multiply()([input_tensor, reshape_output])
    
    return scaled_output

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32x3
    branch1 = block(input_layer)
    branch2 = block(input_layer)

    # Concatenate the outputs from both branches
    concatenated = Concatenate()([branch1, branch2])
    
    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10
    
    model = Model(inputs=input_layer, outputs=output_layer)

    return model