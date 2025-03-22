import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Multiply

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Parallel Convolutional Branches
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine branches via addition
    combined_branches = Add()([branch1, branch2])
    
    # Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(combined_branches)
    
    # Fully Connected Layers for Attention Mechanism
    attention_weights = Dense(units=64, activation='relu')(global_avg_pool)
    attention_weights = Dense(units=2, activation='softmax')(attention_weights)
    
    # Split attention weights for the branches
    weight1, weight2 = attention_weights[:, 0], attention_weights[:, 1]
    
    # Multiply each branch's output by its corresponding weight
    weighted_branch1 = Multiply()([combined_branches, weight1])
    weighted_branch2 = Multiply()([combined_branches, weight2])
    
    # Add the weighted branches
    weighted_output = Add()([weighted_branch1, weighted_branch2])
    
    # Fully Connected Layer for Final Classification
    output_layer = Dense(units=10, activation='softmax')(weighted_output)
    
    # Construct the Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model