import keras
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, multiply, Flatten, Dense
from keras.layers.merge import Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    
    # Parallel paths for channel attention
    avg_pool = GlobalAveragePooling2D()(x)
    max_pool = GlobalMaxPooling2D()(x)
    
    avg_fc1 = Dense(units=128, activation='relu')(avg_pool)
    avg_fc2 = Dense(units=64, activation='relu')(avg_fc1)
    
    max_fc1 = Dense(units=128, activation='relu')(max_pool)
    max_fc2 = Dense(units=64, activation='relu')(max_fc1)
    
    # Add paths together and generate attention weights
    add_layer = Add()([avg_fc2, max_fc2])
    attention = Dense(units=1, activation='sigmoid')(add_layer)
    
    # Element-wise multiplication of features and attention weights
    x = multiply([x, attention])
    
    # Separate average and max pooling for spatial attention
    avg_pool = GlobalAveragePooling2D()(x)
    max_pool = GlobalMaxPooling2D()(x)
    
    # Concatenate spatial and channel features
    concat = Concatenate()([avg_pool, max_pool])
    
    # Fully connected layer for classification
    output = Dense(units=10, activation='softmax')(concat)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Instantiate the model
model = dl_model()
model.summary()