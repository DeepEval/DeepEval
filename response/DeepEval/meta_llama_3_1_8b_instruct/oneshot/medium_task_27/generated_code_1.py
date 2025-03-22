import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense
from keras import regularizers
from keras.optimizers import Adam

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional branch with 3x3 kernel
    branch_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_3x3)
    
    # Convolutional branch with 5x5 kernel
    branch_5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch_5x5)
    
    # Combine the outputs of the two branches
    combined = Add()([branch_3x3, branch_5x5])
    
    # Global average pooling
    global_pool = GlobalAveragePooling2D()(combined)
    
    # Two fully connected layers
    dense1 = Dense(units=128, activation='relu')(global_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Weighted output
    weighted_output = Dense(units=10, activation='softmax')(dense2)
    
    # Attention weights
    attention_weights = Dense(units=10, activation='softmax')(dense2)
    
    # Multiply the weighted output by the attention weights
    weighted_output *= attention_weights
    
    # Final classification
    output_layer = Dense(units=10, activation='softmax')(weighted_output)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer=Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model