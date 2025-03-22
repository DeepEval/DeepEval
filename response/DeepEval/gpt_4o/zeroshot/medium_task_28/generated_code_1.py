import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Softmax, Multiply, LayerNormalization, ReLU, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # 1x1 convolution to generate attention weights
    attention_weights = Conv2D(filters=3, kernel_size=(1, 1), padding='same')(input_layer)
    attention_weights = Softmax(axis=-1)(attention_weights)
    
    # Apply attention weights to input features
    attention_output = Multiply()([input_layer, attention_weights])
    
    # Reduce dimensionality to one-third using 1x1 convolution
    reduced_dim = Conv2D(filters=int(3 / 3), kernel_size=(1, 1), padding='same')(attention_output)
    reduced_dim = LayerNormalization()(reduced_dim)
    reduced_dim = ReLU()(reduced_dim)
    
    # Restore dimensionality using 1x1 convolution
    restored_dim = Conv2D(filters=3, kernel_size=(1, 1), padding='same')(reduced_dim)
    
    # Add processed output to the original input image
    added_output = Add()([input_layer, restored_dim])
    
    # Flatten the output
    flat_output = Flatten()(added_output)
    
    # Fully connected layer for classification
    classification_output = Dense(units=10, activation='softmax')(flat_output)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=classification_output)
    
    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()