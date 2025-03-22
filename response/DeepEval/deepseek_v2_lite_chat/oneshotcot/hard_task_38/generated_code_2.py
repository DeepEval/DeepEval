import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    def block(input_tensor):
        # First pathway
        path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path1 = BatchNormalization()(path1)
        path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
        path1 = BatchNormalization()(path1)
        
        # Second pathway
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = BatchNormalization()(path2)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = BatchNormalization()(path2)

        return Concatenate(axis=-1)([path1, path2])

    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First pathway
    pathway1_output = block(input_tensor=input_layer)
    pathway1_output = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(pathway1_output)
    pathway1_output = Flatten()(pathway1_output)
    pathway1_dense = Dense(units=128, activation='relu')(pathway1_output)
    
    # Second pathway
    pathway2_output = block(input_tensor=input_layer)
    pathway2_output = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(pathway2_output)
    pathway2_output = Flatten()(pathway2_output)
    pathway2_dense = Dense(units=128, activation='relu')(pathway2_output)
    
    # Concatenate outputs from both pathways
    merged_output = Concatenate()([pathway1_output, pathway2_output])
    
    # Fully connected layers
    output_layer = Dense(units=128, activation='relu')(merged_output)
    output_layer = Dense(units=10, activation='softmax')(output_layer)
    
    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])