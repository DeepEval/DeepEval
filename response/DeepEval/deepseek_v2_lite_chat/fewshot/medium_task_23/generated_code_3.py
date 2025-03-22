import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Concatenate, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: Single 1x1 convolution
    conv1_1 = Conv2D(filters=32, kernel_size=1, activation='relu')(input_layer)
    
    # Path 2: 1x1 convolution -> 1x7, 7x1 convolutions
    conv2_1 = Conv2D(filters=64, kernel_size=1, activation='relu')(input_layer)
    conv2_2 = Conv2D(filters=64, kernel_size=(1, 7), padding='same')(conv2_1)
    conv2_3 = Conv2D(filters=64, kernel_size=(7, 1), padding='same')(conv2_1)
    
    # Path 3: 1x1 convolution -> 2 sets of 1x7, 7x1 convolutions
    conv3_1 = Conv2D(filters=96, kernel_size=1, activation='relu')(input_layer)
    conv3_2 = Conv2D(filters=96, kernel_size=(1, 7), padding='same')(conv3_1)
    conv3_3 = Conv2D(filters=96, kernel_size=(7, 1), padding='same')(conv3_1)
    
    # Path 4: Average pooling -> 1x1 convolution
    avg_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    conv4_1 = Conv2D(filters=64, kernel_size=1, activation='relu')(avg_pool)
    
    # Concatenate the outputs
    concat = Concatenate()([conv1_1, conv2_1, conv3_1, conv4_1])
    
    # Flatten and fully connected layer
    flatten = Flatten()(concat)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])