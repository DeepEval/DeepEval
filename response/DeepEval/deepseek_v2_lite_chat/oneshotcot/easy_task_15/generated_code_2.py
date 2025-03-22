import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D
from keras.models import Model

def dl_model():
    # Input Layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    def block1():
        # 3x3 Convolution
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
        # 1x1 Convolution
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(conv1)
        # Average Pooling
        avg_pool = AveragePooling2D(pool_size=(2, 2))(conv2)
        # Dropout Layer
        dropout = keras.layers.Dropout(rate=0.25)(avg_pool)
        
        return dropout
    
    # Block 2
    def block2():
        # 3x3 Convolution
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(block1())
        # 1x1 Convolution
        conv4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(block1())
        # Concatenate Layers
        concat = Concatenate()([conv3, conv4])
        # Batch Normalization
        batch_norm = BatchNormalization()(concat)
        # Flatten Layer
        flatten = Flatten()(batch_norm)
        # Dense Layer
        dense1 = Dense(units=128, activation='relu')(flatten)
        # Dense Layer
        dense2 = Dense(units=64, activation='relu')(dense1)
        # Dense Layer
        output_layer = Dense(units=10, activation='softmax')(dense2)
        
        return output_layer
    
    model = Model(inputs=input_layer, outputs=block2())
    
    return model