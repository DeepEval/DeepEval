import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Add initial 1x1 convolutional layer
    initial_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Define three branches
    
    # Branch 1: Local feature extraction with two 3x3 conv layers
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    
    # Branch 2: Average pooling, 3x3 conv, transposed conv
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(initial_conv)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=2, padding='same', activation='relu')(branch2)
    
    # Branch 3: Average pooling, 3x3 conv, transposed conv
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(initial_conv)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=2, padding='same', activation='relu')(branch3)
    
    # Step 4: Concatenate outputs from the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Step 5: Refine with a 1x1 convolutional layer
    refined = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    
    # Step 6: Flatten the result
    flatten_layer = Flatten()(refined)
    
    # Step 7: Add fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model