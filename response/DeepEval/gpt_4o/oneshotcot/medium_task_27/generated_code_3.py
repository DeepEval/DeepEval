import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 3x3 Convolution
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 2: 5x5 Convolution
    branch2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Element-wise addition of two branches
    added_branches = Add()([branch1, branch2])
    
    # Global Average Pooling
    gap = GlobalAveragePooling2D()(added_branches)
    
    # Attention mechanism to generate weights
    attention_layer1 = Dense(units=64, activation='relu')(gap)
    attention_weights = Dense(units=2, activation='softmax')(attention_layer1)
    
    # Splitting attention weights for branches
    weight1 = attention_weights[:, 0]
    weight2 = attention_weights[:, 1]
    
    # Expanding weights dimensions to match feature maps
    weight1_expanded = weight1[:, None, None, None]
    weight2_expanded = weight2[:, None, None, None]
    
    # Multiply branch outputs by their respective weights
    weighted_branch1 = Multiply()([branch1, weight1_expanded])
    weighted_branch2 = Multiply()([branch2, weight2_expanded])
    
    # Weighted sum of the branches
    weighted_output = Add()([weighted_branch1, weighted_branch2])
    
    # Final fully connected layer for classification
    global_pool = GlobalAveragePooling2D()(weighted_output)
    output_layer = Dense(units=10, activation='softmax')(global_pool)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model