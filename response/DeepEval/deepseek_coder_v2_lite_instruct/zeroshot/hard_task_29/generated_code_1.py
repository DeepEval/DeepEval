import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First block
    # Main path
    main_path = Conv2D(32, (3, 3), activation='relu')(input_layer)
    main_path = Conv2D(32, (1, 1), activation='relu')(main_path)
    
    # Branch path
    branch_path = input_layer
    
    # Combine outputs from both paths
    combined = Add()([main_path, branch_path])
    
    # Second block
    # Pooling layers
    pool1 = MaxPooling2D((1, 1))(combined)
    pool2 = MaxPooling2D((2, 2))(combined)
    pool3 = MaxPooling2D((4, 4))(combined)
    
    # Flatten the results
    flattened1 = Flatten()(pool1)
    flattened2 = Flatten()(pool2)
    flattened3 = Flatten()(pool3)
    
    # Concatenate the flattened results
    concatenated = tf.keras.layers.Concatenate()([flattened1, flattened2, flattened3])
    
    # Fully connected layers
    fc1 = Dense(128, activation='relu')(concatenated)
    fc2 = Dense(10, activation='softmax')(fc1)
    
    # Define the model
    model = Model(inputs=input_layer, outputs=fc2)
    
    return model

# Create the model
model = dl_model()
model.summary()