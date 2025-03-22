import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Block 1
    def block1(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        avg_pool_fc = Dense(units=512, activation='relu')(avg_pool)
        avg_pool_output = Dense(units=128, activation='relu')(avg_pool_fc)
        
        max_pool = GlobalMaxPooling2D()(input_tensor)
        max_pool_fc = Dense(units=512, activation='relu')(max_pool)
        max_pool_output = Dense(units=128, activation='relu')(max_pool_fc)
        
        concat = Concatenate()(inputs=[avg_pool_output, max_pool_output])
        activation = Dense(units=64, activation='relu')(concat)
        return activation
    
    block1_output = block1(conv1)
    
    # Block 2
    def block2(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        avg_pool_fc = Dense(units=256, activation='relu')(avg_pool)
        avg_pool_output = Dense(units=64, activation='relu')(avg_pool_fc)
        
        max_pool = GlobalMaxPooling2D()(input_tensor)
        max_pool_fc = Dense(units=256, activation='relu')(max_pool)
        max_pool_output = Dense(units=64, activation='relu')(max_pool_fc)
        
        concat = Concatenate()(inputs=[avg_pool_output, max_pool_output])
        activation = Dense(units=32, activation='relu')(concat)
        return activation
    
    block2_output = block2(conv1)
    
    # Final concatenation and classification
    concat = Concatenate()([block1_output, block2_output])
    dense1 = Dense(units=256, activation='relu')(concat)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

model = dl_model()
model.summary()