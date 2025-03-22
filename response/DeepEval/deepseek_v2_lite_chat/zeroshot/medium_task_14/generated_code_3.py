from keras.applications import VGG16
from keras.layers import Input, Dense, Flatten, Concatenate, MaxPooling2D, GlobalMaxPooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam


input_shape = (32, 32, 3)


def dl_model():
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First block
    x1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(32, (3, 3), activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D()(x1)
    
    # Second block
    x2 = Conv2D(64, (3, 3), activation='relu')(x1)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(64, (3, 3), activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D()(x2)
    
    # Third block
    x3 = Conv2D(64, (3, 3), activation='relu')(x2)
    x3 = BatchNormalization()(x3)
    x3 = Conv2D(64, (3, 3), activation='relu')(x3)
    x3 = BatchNormalization()(x3)
    x3 = MaxPooling2D()(x3)
    
    # Fourth block (parallel branch)
    x4 = Conv2D(32, (3, 3), activation='relu')(inputs)
    x4 = BatchNormalization()(x4)
    x4 = GlobalMaxPooling2D()(x4)
    
    # Dropout layer
    x4 = Dropout(rate=0.5)(x4)
    
    # Seventh block
    x5 = Flatten()(x4)
    x5 = Dense(512, activation='relu')(x5)
    x5 = Dropout(rate=0.5)(x5)
    
    # Eighth block
    outputs = Dense(10, activation='softmax')(x5)
    
    # Model construction
    model = Model(inputs=inputs, outputs=outputs)
    
    return model