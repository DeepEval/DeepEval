import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Average Pooling Layer
    avg_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(input_layer)
    
    # 1x1 Convolution Layer
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(avg_pool)
    
    # Flatten Layer
    flatten = Flatten()(conv1)
    
    # Dropout Layer
    dropout = Dropout(rate=0.5)(flatten)
    
    # Two Fully Connected Layers
    dense1 = Dense(units=64, activation='relu')(dropout)
    dense2 = Dense(units=32, activation='relu')(dense1)
    
    # Output Layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()
model.summary()