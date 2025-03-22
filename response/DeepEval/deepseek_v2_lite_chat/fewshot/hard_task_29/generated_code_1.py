import keras
from keras.layers import Input, Conv2D, Add, Concatenate, MaxPool2D, Flatten, Dense

def dl_model():
    # First block
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv2)

    # Branch path
    branch_input = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    branch_input = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch_input)
    
    # Second block
    concat = Concatenate()([pool1, branch_input])
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(concat)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(conv3)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv4)
    pool3 = MaxPool2D(pool_size=(4, 4))(conv4)
    
    # Flatten and fully connected layers
    flat1 = Flatten()(pool2)
    flat2 = Flatten()(pool3)
    concat_flat = Concatenate()([flat1, flat2])
    dense1 = Dense(units=512, activation='relu')(concat_flat)
    output = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=[input_layer, branch_input], outputs=output)
    
    return model

model = dl_model()
model.summary()