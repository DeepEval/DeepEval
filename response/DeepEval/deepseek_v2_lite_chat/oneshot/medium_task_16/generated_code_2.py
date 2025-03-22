import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups along the channel dimension
    split_1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Apply 1x1 convolutions to each group independently
    conv_1 = [Conv2D(filters=int(input_layer.shape[3]/3), kernel_size=(1, 1), activation='relu')(group) for group in split_1]
    
    # Average pooling for downsampling
    pool_1 = [MaxPooling2D(pool_size=(2, 2))(conv) for conv in conv_1]
    
    # Concatenate the feature maps along the channel dimension
    concat = Concatenate()(pool_1)
    
    # Flatten and pass through two fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer with softmax activation
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])