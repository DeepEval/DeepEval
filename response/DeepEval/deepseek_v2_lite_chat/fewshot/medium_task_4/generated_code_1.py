import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Add, Flatten, Dense, ZeroPadding2D

def dl_model():
    # Path 1: Convolutional and pooling layers
    input_layer = Input(shape=(32, 32, 3))  # Input shape should match the input images
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
    
    def block1(x):
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
        return MaxPooling2D(pool_size=(2, 2))(x)
    
    pool1_output = block1(pool1)
    
    # Path 2: Single convolutional layer
    def block2(x):
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
        return Flatten()(x)
    
    flat2_output = block2(pool1_output)
    
    # Addition of both pathways
    add_layer = Add()([flat2_output, pool1_output])
    
    # Fully connected layer for classification
    dense = Dense(units=128, activation='relu')(add_layer)
    output = Dense(units=10, activation='softmax')(dense)
    
    # Construct the model
    model = Model(inputs=[input_layer, pool1_output], outputs=output)
    
    return model

# Instantiate the model
model = dl_model()
model.summary()