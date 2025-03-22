import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, GlobalAveragePooling2D, Dense, Flatten

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        avg_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv2)
        return Add()([input_tensor, avg_pool])
    
    block1_output = block1(input_layer)
    
    # Block 2
    def block2(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=512, activation='relu')(avg_pool)
        dense2 = Dense(units=256, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        return output_layer
    
    block2_output = block2(block1_output)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=block2_output)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])