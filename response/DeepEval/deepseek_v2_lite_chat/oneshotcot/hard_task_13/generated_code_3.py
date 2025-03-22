import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, BatchNormalization, GlobalAveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Four parallel branches
    def block1(input_tensor):
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)
        return Concatenate()(inputs=[branch1, branch2, branch3, branch4])
    
    # Block 2: Dimensionality reduction using global average pooling
    def block2(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        return Dense(units=128, activation='relu')(avg_pool)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=block2(block1(input_layer)))
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])