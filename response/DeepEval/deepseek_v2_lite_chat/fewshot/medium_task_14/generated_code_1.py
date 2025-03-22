import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, ReLU, Flatten, Add, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    block1_output = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    block1_output = BatchNormalization()(block1_output)
    block1_output = MaxPooling2D()(block1_output)
    block1_output = Flatten()(block1_output)
    
    # Block 2
    block2_output = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    block2_output = BatchNormalization()(block2_output)
    block2_output = MaxPooling2D()(block2_output)
    block2_output = Flatten()(block2_output)
    
    # Block 3
    block3_output = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    block3_output = BatchNormalization()(block3_output)
    block3_output = MaxPooling2D()(block3_output)
    block3_output = Flatten()(block3_output)
    
    # Parallel branch
    branch_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch_output = BatchNormalization()(branch_output)
    branch_output = MaxPooling2D()(branch_output)
    branch_output = Flatten()(branch_output)
    
    # Concatenate and classify
    concatenated_output = Concatenate()([block1_output, block2_output, block3_output, branch_output])
    dense1 = Dense(units=512, activation='relu')(concatenated_output)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()