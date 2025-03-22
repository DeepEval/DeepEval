import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D, ZeroPadding2D
from keras.models import Model
from keras.regularizers import l2

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    def block1():
        # Path 1: 1x1 max pooling
        path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
        # Path 2: 2x2 max pooling
        path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
        # Path 3: 4x4 max pooling
        path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
        
        # Flatten and regularize paths with dropout
        path1_dropout = Dropout(rate=0.5)(path1)
        path2_dropout = Dropout(rate=0.5)(path2)
        path3_dropout = Dropout(rate=0.5)(path3)
        
        # Concatenate all paths
        concatenated = Concatenate(axis=-1)([path1_dropout, path2_dropout, path3_dropout])
        
        return concatenated
    
    block1_output = block1()
    
    # Block 2
    def block2():
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(block1_output)
        # Path 2: 1x1 convolution + 1x7 convolution + 7x1 convolution
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(block1_output)
        path2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(path2)
        path2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(path2)
        # Path 3: 1x1 convolution + 7x1 convolution + 1x7 convolution
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(block1_output)
        path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(path3)
        path3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(path3)
        # Path 4: Average pooling
        path4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(path4)
        
        # Concatenate all paths
        concatenated = Concatenate(axis=-1)([path1, path2, path3, path4])
        
        return concatenated
    
    block2_output = block2()
    
    # Fully connected layers for final classification
    dense1 = Dense(units=128, activation='relu')(block2_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the deep learning model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])