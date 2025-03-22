from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, concatenate, Dense, Reshape
from keras.models import Model
from keras.layers import InputSpec

def dl_model():
    # Block 1: Feature Extraction and Regularization
    input_layer = Input(shape=(28, 28, 1))
    feature_paths = [
        input_layer,
        Conv2D(32, (1, 1), activation='relu')(input_layer),
        Conv2D(32, (2, 2), activation='relu')(input_layer),
        Conv2D(32, (4, 4), activation='relu')(input_layer)
    ]
    
    maxpool_results = [MaxPooling2D((1, 1), strides=(1, 1))(x) for x in feature_paths]
    flat_results = [Flatten()(x) for x in maxpool_results]
    dropout_results = [Dropout(0.25)(x) for x in concatenate(flat_results)]
    
    block1_output = concatenate(dropout_results)
    
    # Block 2: Multi-Scale Feature Combination
    input_to_block2 = Input(shape=(128,))
    block2_output = block2_output = [
        Dense(256, activation='relu')(input_to_block2),
        Dense(256, activation='relu')(input_to_block2),
        Dense(128, activation='relu')(input_to_block2),
        Dense(64, activation='relu')(input_to_block2),
        Reshape((64,))((input_to_block2))
    ]
    
    # Create the model
    model = Model(inputs=input_layer, outputs=block2_output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Instantiate and return the model
model = dl_model()