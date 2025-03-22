import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First block: three convolutional layers followed by max pooling
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1_1)
    conv1_3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1_2)
    max_pool_1 = MaxPooling2D(pool_size=(2, 2))(conv1_3)
    
    # Second block: four convolutional layers followed by max pooling
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(max_pool_1)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv2_1)
    conv2_3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv2_2)
    conv2_4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv2_3)
    max_pool_2 = MaxPooling2D(pool_size=(2, 2))(conv2_4)

    # Flatten the output
    flatten_layer = Flatten()(max_pool_2)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model