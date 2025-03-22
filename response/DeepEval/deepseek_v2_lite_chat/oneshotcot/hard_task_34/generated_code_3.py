import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    def block(input_tensor):
        # Main Path
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(maxpool1)
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
        
        # Branch Path
        branch_conv = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        maxpool_branch = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(branch_conv)
        
        # Concatenate and continue processing
        concat_layer = Concatenate(axis=-1)([conv2, maxpool2, branch_conv, maxpool_branch])
        
        # Add more layers
        batchnorm = BatchNormalization()(concat_layer)
        flatten = Flatten()(batchnorm)
        dense1 = Dense(units=128, activation='relu')(flatten)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        
        # Construct the model
        model = keras.Model(inputs=input_tensor, outputs=output_layer)
        return model

    # Input layer for the MNIST dataset
    input_layer = Input(shape=(28, 28, 1))
    
    # Model construction
    model = block(input_layer)
    model = keras.Model(inputs=input_layer, outputs=model)

    return model

# Creating the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])