from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Encapsulate tf.split within Lambda layer
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(inputs)
    
    # Convolutional layers for each group
    x1 = Conv2D(32, (1, 1), activation='relu')(x[0])
    x2 = Conv2D(32, (3, 3), activation='relu')(x[1])
    x3 = Conv2D(32, (5, 5), activation='relu')(x[2])
    
    # Dropout layer to mitigate overfitting
    x1 = Dropout(0.2)(x1)
    x2 = Dropout(0.2)(x2)
    x3 = Dropout(0.2)(x3)
    
    # Concatenate the output from each group
    x = Concatenate()([x1, x2, x3])
    
    # Branch pathway
    branch = Conv2D(32, (1, 1), activation='relu')(x)
    
    # Add the output from the branch pathway
    x = Add()([x, branch])
    
    # Flatten and fully connected layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=inputs, outputs=x)
    
    return model