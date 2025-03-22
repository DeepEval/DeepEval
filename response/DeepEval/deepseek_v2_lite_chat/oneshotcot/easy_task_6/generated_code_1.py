import keras
from keras.layers import Input, Conv2D, Add, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Path 1: Main path
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    batch_norm1 = BatchNormalization()(conv2)
    flatten1 = Flatten()(batch_norm1)
    
    # Path 2: Branch path
    branch_input = Input(shape=(28, 28, 1))
    branch_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch_input)
    batch_norm2 = BatchNormalization()(branch_conv)
    flatten2 = Flatten()(batch_norm2)
    
    # Combine the two paths
    combined = Add()([flatten1, flatten2])
    batch_norm2 = BatchNormalization()(combined)
    dense1 = Dense(units=128, activation='relu')(batch_norm2)
    output = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=[input_layer, branch_input], outputs=output)
    
    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])