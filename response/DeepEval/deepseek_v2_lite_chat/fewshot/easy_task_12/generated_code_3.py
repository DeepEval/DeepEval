import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, ReLU

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    def block1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=ReLU)(input_tensor)
        pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)
        return pool1
    
    def block2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=ReLU)(input_tensor)
        pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)
        return pool2
    
    main_path = block1(input_layer)
    branch_path = block2(input_layer)
    
    # Branch path
    def block3(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation=ReLU)(input_tensor)
        return conv1
    
    branch_path = block3(branch_path)
    
    # Sum outputs
    merged = Add()([main_path, branch_path])
    
    # Flattening layer
    flattened = Flatten()(merged)
    
    # Fully connected layer
    output = Dense(units=10, activation='softmax')(flattened)
    
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model

# Construct the model
model = dl_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])