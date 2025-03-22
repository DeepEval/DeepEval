import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, multiply, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv = Conv2D(32, (3, 3), activation='relu')(input_layer)
    
    # Path 1: Global average pooling followed by two fully connected layers
    avgpool1 = GlobalAveragePooling2D()(conv)
    fc1 = Dense(128, activation='relu')(avgpool1)
    fc2 = Dense(64, activation='relu')(fc1)
    
    # Path 2: Global max pooling followed by two fully connected layers
    maxpool1 = GlobalMaxPooling2D()(conv)
    fc1 = Dense(128, activation='relu')(maxpool1)
    fc2 = Dense(64, activation='relu')(fc1)
    
    # Add and multiply the outputs
    add = Add()([fc1, fc2])
    mul = multiply([add, fc1, fc2])
    
    # Global average and max pooling
    avg_pool = GlobalAveragePooling2D()(conv)
    max_pool = GlobalMaxPooling2D()(conv)
    
    # Concatenate and pass through fully connected layer
    concat = Concatenate()([avg_pool, max_pool])
    output = Dense(10, activation='softmax')(concat)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

return model