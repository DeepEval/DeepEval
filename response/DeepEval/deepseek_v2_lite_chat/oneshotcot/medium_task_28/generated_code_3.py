import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, LayerNormalization, Activation

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Attention weights
    attn_weights = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(input_layer)
    
    # Weighted processing with attention weights
    weighted_input = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='linear',
                             kernel=attn_weights)(input_layer)
    
    # Reduce dimensionality and apply layer normalization
    reduced_input = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='linear')(input_layer)
    layn_norm = LayerNormalization()(reduced_input)
    relu = Activation('relu')(layn_norm)
    
    # Expand dimensionality
    expanded_input = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='linear')(relu)
    
    # Add expanded input to original
    final_output = keras.layers.Add()([expanded_input, input_layer])
    
    # Flatten layer
    flatten_layer = Flatten()(final_output)
    
    # Fully connected layers
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dense2 = Dense(units=256, activation='relu')(dense1)
    dense3 = Dense(units=128, activation='relu')(dense2)
    output_layer = Dense(units=10, activation='softmax')(dense3)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

model = dl_model()
model.summary()