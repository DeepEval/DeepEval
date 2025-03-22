import keras
from keras.layers import Input, Conv2D, Multiply, LayerNormalization, Flatten, Dense
from keras.models import Model
import keras.backend as K

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Step 1: Generate attention weights
    attention_weights = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='softmax')(input_layer)
    
    # Step 2: Multiply input features with attention weights
    weighted_features = Multiply()([input_layer, attention_weights])
    
    # Step 3: Dimensionality reduction using 1x1 convolution
    reduced_dimensionality = Conv2D(filters=9, kernel_size=(1, 1), padding='same')(weighted_features)  # 3 channels to 9
    
    # Step 4: Layer normalization and ReLU activation
    norm_layer = LayerNormalization()(reduced_dimensionality)
    activated_layer = keras.layers.ReLU()(norm_layer)
    
    # Step 5: Restore dimensionality with another 1x1 convolution
    restored_dimensionality = Conv2D(filters=3, kernel_size=(1, 1), padding='same')(activated_layer)  # 9 back to 3
    
    # Step 6: Add processed output to the original input image
    added_output = keras.layers.Add()([input_layer, restored_dimensionality])
    
    # Step 7: Flatten and create fully connected layer for classification
    flatten_layer = Flatten()(added_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)  # 10 classes for CIFAR-10
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model