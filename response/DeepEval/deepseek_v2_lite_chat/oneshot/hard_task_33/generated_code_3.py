import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Branch 1
    def branch1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        max_pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(conv1_output)
        return max_pool1
    
    branch1_output = branch1(input_tensor=input_layer)
    
    # Branch 2
    def branch2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        max_pool2 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(conv2)
        return max_pool2
    
    branch2_output = branch2(input_tensor=input_layer)
    
    # Branch 3
    def branch3(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        max_pool3 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(conv2)
        return max_pool3
    
    branch3_output = branch3(input_tensor=input_layer)
    
    # Concatenate outputs from all branches
    concatenated_output = Concatenate(axis=-1)([branch1_output, branch2_output, branch3_output])
    
    # Flatten and fully connected layers
    flatten = Flatten()(concatenated_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])