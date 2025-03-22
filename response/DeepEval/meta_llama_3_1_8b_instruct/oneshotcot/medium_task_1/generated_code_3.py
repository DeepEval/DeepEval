import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Add convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Add convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Step 4: Add maxpooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Step 5: Add input layer to the output of maxpooling layer
    added_layer = Add()([input_layer, max_pooling])
    
    # Step 6: Add flatten layer
    flatten_layer = Flatten()(added_layer)
    
    # Step 7: Add dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Step 8: Add dense layer
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Compile the model (optional)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])