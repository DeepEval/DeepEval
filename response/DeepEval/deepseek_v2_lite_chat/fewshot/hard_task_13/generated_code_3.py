import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Concatenate, Dense, GlobalAveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First block of layers
    def block1(input_tensor):
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1_1)
        conv1_3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(conv1_2)
        avg_pool1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(conv1_3)
        
        return Concatenate()([conv1_3, avg_pool1])
    
    # Second block of layers
    def block2(input_tensor):
        avg_pool2 = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=128, activation='relu')(avg_pool2)
        dense2 = Dense(units=64, activation='relu')(dense1)
        
        # Reshape dense2 to match the input shape of block1
        reshape = Reshape((4, 4, 8))(dense2)
        # Multiply element-wise with the input feature map
        multiply = keras.layers.Multiply()([input_tensor, reshape])
        dense3 = Dense(units=32, activation='relu')(multiply)
        output_layer = Dense(units=10, activation='softmax')(dense3)
        
        return output_layer
    
    # First block output
    block1_output = block1(input_tensor=input_layer)
    # Second block output
    model = block2(input_tensor=block1_output)
    
    return model

# Instantiate and return the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])