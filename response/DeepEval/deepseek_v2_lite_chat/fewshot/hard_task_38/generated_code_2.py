import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Pathway 1
    def pathway_1(input_tensor):
        # Repeated block structure
        for _ in range(3):
            conv2d = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
            batch_norm = BatchNormalization()(conv2d)
            maxpool = MaxPooling2D(pool_size=(2, 2))(batch_norm)
        
        # Concatenate feature maps from repeated blocks
        concat_layer = Concatenate()([maxpool, batch_norm, maxpool, batch_norm])

        return concat_layer

    pathway1_output = pathway_1(input_tensor=input_layer)

    # Pathway 2
    def pathway_2(input_tensor):
        # Repeated block structure
        for _ in range(3):
            conv2d = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
            batch_norm = BatchNormalization()(conv2d)
            maxpool = MaxPooling2D(pool_size=(2, 2))(batch_norm)
        
        # Concatenate feature maps from repeated blocks
        concat_layer = Concatenate()([maxpool, batch_norm, maxpool, batch_norm])

        return concat_layer

    pathway2_output = pathway_2(input_tensor=input_layer)

    # Merge outputs from both pathways
    merged_output = Concatenate()([pathway1_output, pathway2_output])

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(merged_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model structure
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()