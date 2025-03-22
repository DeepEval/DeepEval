import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, SeparableConv2D
from keras.models import Model
from keras.layers import Add

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='linear')(conv1)
    dropout1 = Dropout(0.5)(conv1)
    
    # Branch path directly connected to inputs
    branch = inputs
    
    # Concatenate outputs from both paths
    output1 = Add()([dropout1, branch])
    
    # Second block
    split1 = Lambda(lambda x: keras.backend.cast(keras.backend.split(x, 3, axis=-1), keras.backend.floatx())).(output1)
    
    conv2_1 = SeparableConv2D(filters=64, kernel_size=(1, 1), activation='relu')(split1[0])
    conv2_2 = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu')(split1[1])
    conv2_3 = SeparableConv2D(filters=64, kernel_size=(5, 5), activation='relu')(split1[2])
    
    dropout2 = Dropout(0.5)(Concatenate()([conv2_1, conv2_2, conv2_3]))
    
    flatten = Flatten()(dropout2)
    
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    outputs = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model