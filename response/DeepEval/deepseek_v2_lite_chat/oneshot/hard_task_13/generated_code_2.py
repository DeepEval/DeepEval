import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First block: four parallel branches
    def block1(input_tensor):
        branch1 = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu', kernel_regularizer='l2')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer='l2')(input_tensor)
        branch3 = Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu', kernel_regularizer='l2')(input_tensor)
        branch4 = MaxPooling2D(pool_size=3, strides=1, padding='same')(input_tensor)
        return Concatenate()(inputs=[branch1, branch2, branch3, branch4])
    
    block1_output = block1(input_tensor=input_layer)
    block1_output = BatchNormalization()(block1_output)
    block1_output = Flatten()(block1_output)
    
    # Second block: dimensionality reduction using global average pooling
    def block2(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        return Dense(units=128, activation='relu')(avg_pool)
    
    block2_output = block2(input_tensor=block1_output)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(block2_output)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model