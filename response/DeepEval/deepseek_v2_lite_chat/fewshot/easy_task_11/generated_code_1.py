import keras
from keras.models import Model
from keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dense, Dropout, Activation

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Average Pooling layer with a 5x5 window and a 3x3 stride
    avg_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(input_layer)
    
    # 1x1 convolutional layer
    conv = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(avg_pool)
    
    # Flatten the feature maps
    flatten = Flatten()(conv)
    
    # Two fully connected layers with dropout for regularization
    dense1 = Dense(units=128, activation='relu')(flatten)
    dropout = Dropout(rate=0.5)(dense1)
    dense2 = Dense(units=64, activation='relu')(dropout)
    
    # Output layer with softmax activation for multi-class classification
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and return the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Model structure:")
model.summary()

return model