import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    # Main path
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    dropout1 = Dropout(0.5)(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(dropout1)
    dropout2 = Dropout(0.5)(conv2)

    # Branch path
    branch_input = input_layer

    # Combine both paths
    combined_output = Add()([dropout2, branch_input])

    # Batch normalization and flattening
    bn_output = BatchNormalization()(combined_output)
    flatten_layer = Flatten()(bn_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=[input_layer, branch_input], outputs=output_layer)

    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])