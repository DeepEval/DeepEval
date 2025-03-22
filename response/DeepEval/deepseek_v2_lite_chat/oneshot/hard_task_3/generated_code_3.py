import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Dropout, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Split the input into three groups
    x = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(inputs)
    
    # Process each group
    for i in range(3):
        # 1x1 convolution
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(x[i])
        # 3x3 convolution
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv1)
        # Dropout layer
        conv2 = Dropout(0.5)(conv2)
        # Batch normalization
        conv2 = BatchNormalization()(conv2)
        # Max pooling
        conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Concatenate outputs from the three groups
    main_path = conv2
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(x[0])(inputs)
    
    # Add the branch pathway output to the main pathway output
    main_path = Concatenate()([main_path, branch_path])
    
    # Flatten and fully connected layers
    main_path = Flatten()(main_path)
    main_path = Dense(units=512, activation='relu')(main_path)
    main_path = Dense(units=10, activation='softmax')(main_path)
    
    # Return the constructed model
    return Model(inputs=inputs, outputs=main_path)

# Create the model
model = dl_model()
model.summary()