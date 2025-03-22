import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, SeparableConv2D
from keras.regularizers import l2
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Branch 1
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    branch1 = SeparableConv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = MaxPooling2D(pool_size=(2, 2))(branch1)
    branch1 = Dense(units=512, activation='relu')(branch1)
    branch1 = Dropout(0.5)(branch1)
    
    # Branch 2
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    branch2 = SeparableConv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2))(branch2)
    branch2 = Dense(units=512, activation='relu')(branch2)
    branch2 = Dropout(0.5)(branch2)
    
    # Branch 3
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    branch3 = SeparableConv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch3)
    branch3 = BatchNormalization()(branch3)
    branch3 = MaxPooling2D(pool_size=(2, 2))(branch3)
    branch3 = Dense(units=512, activation='relu')(branch3)
    branch3 = Dropout(0.5)(branch3)
    
    # Concatenate the outputs of the three branches
    concat = Concatenate()([branch1, branch2, branch3])
    
    # Fully connected layers
    fc1 = Dense(units=256, activation='relu', kernel_regularizer=l2(0.001))(concat)
    fc2 = Dense(units=128, activation='relu', kernel_regularizer=l2(0.001))(fc1)
    output = Dense(units=10, activation='softmax')(fc2)
    
    # Model
    model = Model(inputs=input_layer, outputs=output)
    
    return model