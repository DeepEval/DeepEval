import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Pathway 1
    def pathway1_block(input_tensor):
        block = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        block = BatchNormalization()(block)
        block = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block)
        block = BatchNormalization()(block)
        return block

    pathway1_output = pathway1_block(input_tensor=input_layer)
    pathway1_output = MaxPooling2D(pool_size=(2, 2))(pathway1_output)

    # Pathway 2
    def pathway2_block(input_tensor):
        block = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        block = BatchNormalization()(block)
        block = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block)
        block = BatchNormalization()(block)
        return block

    pathway2_output = pathway2_block(input_tensor=input_layer)
    pathway2_output = MaxPooling2D(pool_size=(2, 2))(pathway2_output)

    # Concatenate Pathways
    concatenated = Concatenate()([pathway1_output, pathway2_output])

    # Flatten and Fully Connected Layers
    flattened = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])