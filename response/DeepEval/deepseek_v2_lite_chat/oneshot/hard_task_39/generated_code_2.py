import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    def block1(input_tensor):
        # Three max pooling layers with different scales
        pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_tensor)
        # Flatten and concatenate the outputs
        flat = Flatten()(Concatenate()([pool1, pool2, pool3]))
        
        # Reshape the output for Block 2
        reshape = Reshape((-1, 16))(flat)
        
        # Fully connected layer and output layer
        dense1 = Dense(128, activation='relu')(reshape)
        output = Dense(10, activation='softmax')(dense1)
        
        return output
    
    # Block 2
    def block2(input_tensor):
        # Four parallel paths
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)
        
        # Concatenate the outputs
        output = Concatenate()([path1, path2, path3, path4])
        
        # Batch normalization and flatten
        batch_norm = BatchNormalization()(output)
        flatten = Flatten()(batch_norm)
        
        # Fully connected layer and output layer
        dense1 = Dense(128, activation='relu')(flatten)
        output = Dense(10, activation='softmax')(dense1)
        
        return output
    
    # Model
    model = keras.Model(inputs=input_layer, outputs=block2(block1(input_layer)))
    
    return model

# Create the deep learning model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])