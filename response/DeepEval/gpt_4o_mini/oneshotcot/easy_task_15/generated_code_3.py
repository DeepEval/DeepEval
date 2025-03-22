import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    def block(input_tensor):
        # 3x3 Convolutional layer
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Two 1x1 Convolutional layers
        conv1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3x3)
        conv1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1x1_1)
        
        # Average Pooling layer
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1x1_2)
        
        # Dropout layer
        dropout = Dropout(0.5)(avg_pool)
        
        return dropout
    
    # First block
    block_output_1 = block(input_layer)
    
    # Second block
    block_output_2 = block(block_output_1)
    
    # Global Average Pooling layer
    global_avg_pool = GlobalAveragePooling2D()(block_output_2)
    
    # Flatten layer
    flatten_layer = Flatten()(global_avg_pool)
    
    # Fully connected layer
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense_layer)
    
    # Constructing the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model