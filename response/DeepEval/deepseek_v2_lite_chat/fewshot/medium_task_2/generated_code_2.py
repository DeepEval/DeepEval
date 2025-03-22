import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Branch path
    branch_conv = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(input_layer)
    branch_pool = MaxPooling2D(pool_size=(2, 2))(branch_conv)
    
    # Concatenate features from both paths
    concat = Add()([pool2, branch_pool])
    
    # Flatten and output layer
    flatten = Flatten()(concat)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    # Model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create and return the model
model = dl_model()
model.summary()