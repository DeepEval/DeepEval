import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Concatenate, BatchNormalization, Flatten, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1
    def branch1():
        # Block 1
        block = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(input_layer)
        block = BatchNormalization()(block)
        block = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(block)
        block = Flatten()(block)
        
        # Fully connected layers
        dense1 = Dense(units=512, activation='relu')(block)
        output1 = Dense(units=128, activation='relu')(dense1)

        return output1

    # Branch 2
    def branch2():
        # Global average pooling
        block = GlobalAveragePooling2D()(input_layer)
        
        # Fully connected layers
        dense1 = Dense(units=512, activation='relu')(block)
        output2 = Dense(units=128, activation='relu')(dense1)

        return output2

    # Output from both branches
    output1 = branch1()
    output2 = branch2()

    # Concatenate the outputs
    concat = Concatenate()([output1, output2])

    # Flatten and fully connected layer
    flatten = Flatten()(concat)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])