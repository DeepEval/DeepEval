import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Add, Multiply, Flatten
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(conv2)
    main_path = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Branch path
    branch = GlobalAveragePooling2D()(input_layer)
    branch = Dense(128, activation='relu')(branch)
    branch = Dense(64, activation='relu')(branch)
    weights = Dense(128, activation='sigmoid')(branch)
    weights = Flatten()(weights)
    weights = Dense(32 * 32 * 3, activation='sigmoid')(weights)
    weights = Reshape((32, 32, 3))(weights)
    weighted_input = Multiply()([weights, input_layer])
    
    # Addition of main path and weighted input
    added = Add()([main_path, weighted_input])
    
    # Additional fully connected layers
    flatten = Flatten()(added)
    dense1 = Dense(128, activation='relu')(flatten)
    output_layer = Dense(10, activation='softmax')(dense1)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model