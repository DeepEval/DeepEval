import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    def block1():
        # 3x3 convolutional layer
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
        # 1x1 convolutional layer
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(conv1)
        # Average pooling layer
        avg_pool = AveragePooling2D(pool_size=(2, 2))(conv2)
        # Dropout layer for regularization
        dropout = Dropout(0.2)(avg_pool)
        
        return dropout
    
    # Block 2
    def block2():
        # 1x1 convolutional layer
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(dropout)
        # 3x3 convolutional layer
        conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(dropout)
        # 1x1 convolutional layer
        conv5 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(dropout)
        # Concatenate layers
        concat = Concatenate()([conv3, conv4, conv5])
        
        return concat
    
    # Apply the blocks
    block1_output = block1()
    block2_output = block2()
    
    # Global average pooling to reduce dimensions
    avg_pool = GlobalAveragePooling2D()(block2_output)
    # Flatten layer
    flatten = Flatten()(avg_pool)
    # Fully connected layer
    dense = Dense(units=128, activation='relu')(flatten)
    # Output layer
    output = Dense(units=10, activation='softmax')(dense)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output)
    
    return model