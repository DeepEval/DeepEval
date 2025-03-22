import keras
from keras.layers import Input, Conv2D, Softmax, Multiply, LayerNormalization, ReLU, Flatten, Dense
from keras.models import Model

def dl_model():
    
    # Step 1: Input Layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)
    
    # Step 2: Generate Attention Weights
    attention_weights = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    attention_weights = Softmax()(attention_weights)  # Apply softmax to get attention weights
    
    # Step 3: Multiply with Input Features
    weighted_features = Multiply()([input_layer, attention_weights])
    
    # Step 4: Dimensionality Reduction
    reduced_dimensionality = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same')(weighted_features)
    layer_norm = LayerNormalization()(reduced_dimensionality)
    relu_activation = ReLU()(layer_norm)
    
    # Step 5: Restore Dimensionality
    restored_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(relu_activation)
    
    # Step 6: Skip Connection (Add processed output to original input)
    final_output = keras.layers.add([input_layer, restored_output])
    
    # Step 7: Flatten and Fully Connected Layer
    flatten_layer = Flatten()(final_output)
    dense_output = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=dense_output)

    return model