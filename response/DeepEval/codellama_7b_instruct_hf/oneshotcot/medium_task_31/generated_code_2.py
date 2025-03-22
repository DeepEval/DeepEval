import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Normalize the pixel values to be between 0 and 1
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255
    
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Define the splitting layer
    splitting_layer = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Define the convolutional layers
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(splitting_layer[0])
    conv1_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(splitting_layer[1])
    conv1_5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(splitting_layer[2])
    
    # Define the concatenate layer
    concatenate_layer = Concatenate()([conv1_1, conv1_3, conv1_5])
    
    # Define the batch normalization layer
    bath_norm = BatchNormalization()(concatenate_layer)
    
    # Define the flatten layer
    flatten_layer = Flatten()(bath_norm)
    
    # Define the fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
    
    return model