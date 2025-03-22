import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Three parallel max pooling layers with different scales
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    
    # Flatten each pooling output
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)
    
    # Concatenate flattened outputs
    concat1 = Concatenate()([flat1, flat2, flat3])
    
    # Fully connected layer and reshape for Block 2
    fc1 = Dense(units=256, activation='relu')(concat1)
    reshaped = Reshape((4, 4, 16))(fc1)  # Reshape according to the dimensions needed for Block 2
    
    # Block 2: Multiple branches with convolution and pooling
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshaped)
    path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshaped)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(reshaped)
    
    # Concatenate outputs from all branches
    concat2 = Concatenate()([path1, path2, path3, path4])
    
    # Final classification layers
    flatten_layer = Flatten()(concat2)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model