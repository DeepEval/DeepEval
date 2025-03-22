import keras
from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    
    # Main path layers
    input_layer = Input(shape=(28, 28, 1))
    conv_block1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    conv_block2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_block1)
    max_pool1 = MaxPooling2D(pool_size=(2, 2))(conv_block2)
    
    # Branch path layers
    branch_input = Conv2D(32, (1, 1), activation='relu', padding='same')(max_pool1)
    branch_output = Conv2D(32, (1, 1), activation='relu')(branch_input)
    
    # Sum main and branch paths
    merged = Concatenate()([max_pool1, branch_output])
    
    # Additional blocks
    batch_norm = BatchNormalization()(merged)
    flatten = Flatten()(batch_norm)
    dense = Dense(512, activation='relu')(flatten)
    output = Dense(10, activation='softmax')(dense)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Instantiate the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()