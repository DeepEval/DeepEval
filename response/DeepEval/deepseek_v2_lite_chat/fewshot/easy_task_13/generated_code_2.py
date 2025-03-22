import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Dropout, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 convolutional layers
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Dropout layer for regularization
    conv1 = Dropout(rate=0.5)(conv1_1)
    conv1 = Add()([conv1_1, conv1_2])
    
    # 3x1 convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 1), activation='relu')(conv1)
    
    # 1x3 convolutional layer
    conv3 = Conv2D(filters=64, kernel_size=(1, 3), activation='relu')(conv1)
    
    # 1x1 convolutional layer to match channels
    conv4 = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(conv3)
    
    # Flatten and fully connected layers
    flatten = Flatten()(conv4)
    dense = Dense(units=128, activation='relu')(flatten)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Creating the model
model = dl_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])