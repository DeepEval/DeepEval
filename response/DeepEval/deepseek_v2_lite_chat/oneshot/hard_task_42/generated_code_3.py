import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape
from keras.layers import Dropout

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    def block1():
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
        conv2 = Conv2D(filters=32, kernel_size=(2, 2), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(4, 4), padding='same', activation='relu')(input_layer)
        maxpool1 = MaxPooling2D(pool_size=(1, 1), padding='same')(conv3)
        
        # Dropout for regularization
        dropout1 = Dropout(0.5)(maxpool1)
        dropout2 = Dropout(0.5)(maxpool1)
        
        # Concatenate and reshape for Block 2
        concat = Concatenate(axis=-1)([dropout1, dropout2])
        reshape = Reshape((-1, 1))(concat)
        
        return reshape
    
    block1_output = block1()
    
    # Block 2
    def block2():
        dense1 = Dense(units=128, activation='relu')(block1_output)
        dense2 = Dense(units=64, activation='relu')(dense1)
        
        # Four parallel paths
        path1 = Dense(units=64, activation='relu')(block1_output)
        path2 = Dense(units=64, activation='relu')(block1_output)
        path3 = Dense(units=64, activation='relu')(block1_output)
        path4 = Dense(units=64, activation='relu')(block1_output)
        
        # Convolution and pooling strategies
        conv1_1x1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(path1)
        conv1_1x7_1x7 = Conv2D(filters=64, kernel_size=(1, 7), activation='relu')(path2)
        conv1_7x1_1x7 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu')(path3)
        conv1_avgpool = MaxPooling2D(pool_size=(1, 1), padding='same')(path4)
        
        # Concatenate and reshape
        concat = Concatenate(axis=-1)([conv1_1x1, conv1_1x7_1x7, conv1_7x1_1x7, conv1_avgpool])
        reshape = Reshape((-1, 1))(concat)
        
        # Fully connected layers for output
        dense3 = Dense(units=128, activation='relu')(reshape)
        output_layer = Dense(units=10, activation='softmax')(dense3)
        
        return output_layer
    
    model = keras.Model(inputs=input_layer, outputs=block2())
    
    return model

# Create and return the deep learning model
model = dl_model()