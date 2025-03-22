import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First branch with 3x3 convolution
    branch1_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Second branch with 5x5 convolution
    branch2_conv = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add the outputs of the two branches
    combined_branches = Add()([branch1_conv, branch2_conv])
    
    # Global Average Pooling
    pooled_features = GlobalAveragePooling2D()(combined_branches)
    
    # Fully connected layers for attention weights
    attention_weights1 = Dense(units=64, activation='relu')(pooled_features)
    attention_weights2 = Dense(units=10, activation='softmax')(attention_weights1)
    
    # Element-wise multiplication with attention weights
    weighted_branch1 = Multiply()([branch1_conv, attention_weights2])
    weighted_branch2 = Multiply()([branch2_conv, attention_weights2])
    
    # Combine weighted branches
    weighted_output = Add()([weighted_branch1, weighted_branch2])
    
    # Flatten and output layer for classification
    flatten_layer = Flatten()(weighted_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model