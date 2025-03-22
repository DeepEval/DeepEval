import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape

def dl_model():
    def block1():
        # Input layer
        input_layer = Input(shape=(28, 28, 1))
        
        # First path
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        
        # Second path
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        
        # Third path
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')(input_layer)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        
        # Max pooling with varying window sizes
        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
        
        # Flatten and concatenate
        flatten = Flatten()(Concatenate()([conv1, conv2, conv3, maxpool1]))
        
        # Fully connected layers
        dense1 = Dense(units=128, activation='relu')(flatten)
        dense2 = Dense(units=64, activation='relu')(dense1)
        
        return keras.Model(inputs=input_layer, outputs=dense2)
    
    def block2():
        # Input layer
        input_layer = Input(shape=(28, 28, 1))
        
        # First branch
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv2)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)
        
        # Concatenate all branches
        concat = Concatenate()([conv1, conv2, conv3, maxpool])
        
        # Batch normalization and flatten
        bn = BatchNormalization()(concat)
        flat = Flatten()(bn)
        
        # Fully connected layers
        dense1 = Dense(units=128, activation='relu')(flat)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        
        return keras.Model(inputs=input_layer, outputs=output_layer)
    
    # Block 1
    block1_model = block1()
    # Block 2
    block2_model = block2()
    
    # Connect Block 1 and Block 2
    model = keras.Model(inputs=[block1_model.input, block2_model.input], outputs=[block1_model.output, block2_model.output])
    
    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])