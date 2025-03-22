import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense

def dl_model():
    # Main path
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Branch path
    branch_input = Input(shape=(28, 28, 1))
    branch_conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(branch_input)
    
    # Add the main path and branch path outputs
    add_layer = Add()([pool1, branch_conv])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(add_layer)
    flatten = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=[input_layer, branch_input], outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])