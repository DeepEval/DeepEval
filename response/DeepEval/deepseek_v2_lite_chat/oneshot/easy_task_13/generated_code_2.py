import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add
from keras.models import Model
from keras.regularizers import l2

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First 1x1 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    # Dropout layer to mitigate overfitting
    conv1 = BatchNormalization()(conv1)
    conv1 = keras.layers.Dropout(0.5)(conv1)
    
    # Second 1x1 convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = BatchNormalization()(conv2)
    conv2 = keras.layers.Dropout(0.5)(conv2)
    
    # 3x1 convolutional layer
    conv3 = Conv2D(filters=32, kernel_size=(3, 1), padding='valid', activation='relu')(conv1)
    conv3 = BatchNormalization()(conv3)
    conv3 = keras.layers.Dropout(0.5)(conv3)
    
    # 1x3 convolutional layer
    conv4 = Conv2D(filters=32, kernel_size=(1, 3), padding='valid', activation='relu')(conv2)
    conv4 = BatchNormalization()(conv4)
    conv4 = keras.layers.Dropout(0.5)(conv4)
    
    # 1x1 convolutional layer to adjust channel count
    conv5 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    conv5 = BatchNormalization()(conv5)
    conv5 = keras.layers.Dropout(0.5)(conv5)
    
    # Add all convolutional layers
    added_layer = Add()([conv3, conv4, conv5])
    
    # Flatten layer
    flatten_layer = Flatten()(added_layer)
    
    # Fully connected layer
    dense1 = Dense(units=128, activation='relu', kernel_regularizer=l2(0.001))(flatten_layer)
    dense1 = BatchNormalization()(dense1)
    dense1 = keras.layers.Dropout(0.5)(dense1)
    
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])