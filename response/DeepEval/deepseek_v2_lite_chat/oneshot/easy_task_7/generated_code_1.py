import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    # Main Path
    input_layer = Input(shape=(28, 28, 1))
    
    # Main Path Conv+Dropout
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    dropout1 = keras.layers.Dropout(0.2)(conv1)
    
    # Main Path Conv2D+BatchNorm
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout1)
    batch_norm1 = BatchNormalization()(conv2)
    
    # Branch Path
    branch_input = Input(shape=(28, 28, 1))
    branch_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_input)
    
    # Combine Paths
    combined = Concatenate()([batch_norm1, branch_conv])
    
    # Flattening Layer and Fully Connected Layers
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=[input_layer, branch_input], outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])