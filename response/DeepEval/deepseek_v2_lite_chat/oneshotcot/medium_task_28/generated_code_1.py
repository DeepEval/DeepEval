import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, LayerNormalization, ReLU

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Attention weights generation
    attention_weights = Conv2D(filters=1, kernel_size=(1, 1), activation='softmax')(input_layer)  # 1x1 convolution
    
    # Weighted processing
    context_vector = keras.layers.multiply([attention_weights, input_layer])  # Multiply attention weights with input features
    reduced_input = Conv2D(filters=int(input_layer.shape[3]/3), kernel_size=(1, 1), activation='relu')(context_vector)  # Reduce dimensionality
    normalized_input = LayerNormalization(axis=-1)(reduced_input)  # Layer normalization
    restored_input = Conv2D(filters=int(input_layer.shape[3]), kernel_size=(1, 1), activation='sigmoid')(normalized_input)  # Restore dimensionality
    
    # Add processed input back to original image
    final_input = keras.layers.add([input_layer, restored_input])
    
    # Flatten layer
    flattened_input = Flatten()(final_input)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened_input)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()