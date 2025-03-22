import keras
from keras.models import Model
from keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dense, Dropout, multiply

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Average Pooling layer with 5x5 window and 3x3 stride
    avg_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(input_layer)
    
    # 1x1 convolutional layer
    conv = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(avg_pool)
    
    # Flatten layer
    flatten = Flatten()(conv)
    
    # Dropout layer for regularization
    dropout = Dropout(rate=0.5)(flatten)
    
    # Fully connected layer 1
    dense1 = Dense(units=128, activation='relu')(dropout)
    
    # Fully connected layer 2
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=dense2)
    
    return model

# Instantiate and return the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

return model