import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block1(input_tensor):
        # First convolutional layer
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        # Average pooling layer
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
        # Second convolutional layer
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
        # Add first block output to main path
        add1 = Add()([input_tensor, conv2])
        return add1
    
    # Block 2
    def block2(input_tensor):
        # Global average pooling
        pool2 = GlobalAveragePooling2D()(input_tensor)
        # Two fully connected layers
        fc1 = Dense(units=128, activation='relu')(pool2)
        # Refine weights using channel weights
        output_weights = Dense(units=128)(fc1)
        # Multiply input with weights
        output = keras.layers.multiply([input_tensor, output_weights])
        # Flatten and add one more dense layer
        flatten = Flatten()(output)
        dense = Dense(units=10, activation='softmax')(flatten)
        # Return concatenated model
        model = keras.Model(inputs=input_layer, outputs=[dense, output])
        return model
    
    # Construct model
    input_tensor = block1(input_layer)
    model = block2(input_tensor)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mse'], loss_weights=[1., 0.2], metrics=['accuracy'])