import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels
    
    # Average pooling layers with different pool sizes and strides
    avg_pool_1x1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    avg_pool_2x2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    avg_pool_4x4 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)
    
    # Flatten each pooled output
    flatten_1x1 = Flatten()(avg_pool_1x1)
    flatten_2x2 = Flatten()(avg_pool_2x2)
    flatten_4x4 = Flatten()(avg_pool_4x4)
    
    # Concatenate the flattened outputs
    concatenated = Concatenate()([flatten_1x1, flatten_2x2, flatten_4x4])
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concatenated)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()