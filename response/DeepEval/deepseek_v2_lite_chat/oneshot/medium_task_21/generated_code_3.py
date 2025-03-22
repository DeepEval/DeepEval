import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.regularizers import l2

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Define four branches
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', kernel_regularizer=l2(0.001))(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001))(input_layer)
    branch4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch3)
    branch5 = AveragePooling2D(pool_size=(3, 3))(input_layer)
    branch6 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch5)
    branch7 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch6)
    branch8 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch7)

    # Apply dropout for regularization
    branch1 = keras.layers.Dropout(0.5)(branch1)
    branch2 = keras.layers.Dropout(0.5)(branch2)
    branch3 = keras.layers.Dropout(0.5)(branch3)
    branch4 = keras.layers.Dropout(0.5)(branch4)
    branch5 = keras.layers.Dropout(0.5)(branch5)
    branch6 = keras.layers.Dropout(0.5)(branch6)
    branch7 = keras.layers.Dropout(0.5)(branch7)
    branch8 = keras.layers.Dropout(0.5)(branch8)

    # Concatenate outputs from all branches
    concat = Concatenate()([branch1, branch2, branch3, branch4, branch5, branch6, branch7, branch8])
    
    # Fully connected layers
    dense1 = Dense(units=512, activation='relu')(concat)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model