import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Dense, Flatten

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    block1_output = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    block1_output = BatchNormalization()(block1_output)
    
    # Block 2
    block2_output = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(block1_output)
    block2_output = BatchNormalization()(block2_output)
    
    # Block 3
    block3_output = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(block2_output)
    block3_output = BatchNormalization()(block3_output)
    
    # Parallel branch
    direct_output = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Combine outputs from blocks and parallel branch
    combined_output = Add()([block3_output, direct_output])
    
    # Flatten and pass through dense layers
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

model = dl_model()
model.summary()