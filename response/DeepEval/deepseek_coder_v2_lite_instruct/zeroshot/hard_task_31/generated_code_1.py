import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dropout, Add, Lambda, SeparableConv2D, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # First block
    # Main path
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    
    # Branch path
    y = Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    
    # Addition of main and branch paths
    x = Add()([x, y])
    
    # Second block
    # Split the input into three groups
    split_layer = Lambda(lambda tensor: tf.split(tensor, num_or_size_splits=3, axis=-1))
    x = split_layer(x)
    
    # Process each group with separable convolutions and dropout
    outputs = []
    for i in range(3):
        if i == 0:
            conv = SeparableConv2D(32, (1, 1), padding='same', activation='relu')(x[i])
        elif i == 1:
            conv = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(x[i])
        elif i == 2:
            conv = SeparableConv2D(32, (5, 5), padding='same', activation='relu')(x[i])
        conv = Dropout(0.2)(conv)
        outputs.append(conv)
    
    # Concatenate the outputs from the three groups
    x = tf.concat(outputs, axis=-1)
    
    # Flatten layer
    x = Flatten()(x)
    
    # Fully connected layer
    x = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=x)
    
    return model

# Example usage
model = dl_model()
model.summary()