import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Concatenate, BatchNormalization, Dropout, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    block1_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1_output = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(block1_output)
    block1_output = Flatten()(block1_output)
    block1_output = Dropout(0.5)(block1_output)
    
    # Between Block 1 and Block 2
    intermediate_layer = Dense(units=128, activation='relu')(block1_output)
    intermediate_layer = Dense(units=4)(intermediate_layer)  # Reshape to 4-dimensional tensor
    
    # Block 2
    block2_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block2_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block2_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block2_output = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(block2_output)
    
    # Branch connections
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    
    # Concatenation
    concatenated_output = Concatenate(axis=-1)([block2_output, branch1, branch2, branch3, branch4])
    concatenated_output = BatchNormalization()(concatenated_output)
    
    # Dense layers
    dense1 = Dense(units=256, activation='relu')(concatenated_output)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

model = dl_model()
model.summary()