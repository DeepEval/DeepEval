import keras
from keras.layers import Input, Conv2D, Multiply, Softmax, LayerNormalization, ReLU, Flatten, Dense
from keras.models import Model

def dl_model():
    
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)
    
    # Step 2: Generate attention weights with a 1x1 convolution
    attention_weights = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    attention_weights = Softmax()(attention_weights)  # Apply softmax to get weights
    
    # Step 3: Multiply weights with input features to obtain contextual information
    context = Multiply()([input_layer, attention_weights])
    
    # Step 4: Reduce dimensionality using another 1x1 convolution
    reduced_dimensionality = Conv2D(filters=10, kernel_size=(1, 1), padding='same')(context)  # Reduce to 10 filters
    reduced_dimensionality = LayerNormalization()(reduced_dimensionality)  # Apply layer normalization
    reduced_dimensionality = ReLU()(reduced_dimensionality)  # Apply ReLU activation
    
    # Step 5: Restore dimensionality with an additional 1x1 convolution
    restored_dimensionality = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(reduced_dimensionality)
    
    # Step 6: Add the processed output to the original input image
    added_output = keras.layers.Add()([input_layer, restored_dimensionality])
    
    # Step 7: Add flatten layer
    flatten_layer = Flatten()(added_output)
    
    # Step 8: Add a fully connected layer for classification
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    
    # Step 9: Output layer for classification (10 classes for CIFAR-10)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model