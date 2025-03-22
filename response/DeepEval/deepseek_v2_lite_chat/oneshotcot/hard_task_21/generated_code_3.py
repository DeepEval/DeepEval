import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups
    channel_split1 = Lambda(lambda x: keras.backend.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    channel_split2 = Lambda(lambda x: keras.backend.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    channel_split3 = Lambda(lambda x: keras.backend.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Feature extraction with depthwise separable convolutional layers
    # Main path: 1x1, 3x3, 5x5
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(channel_split1[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(channel_split2[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(channel_split3[2])
    
    # Concatenate the outputs from the main path
    main_output = Concatenate()([conv1, conv2, conv3])
    
    # Branch path: 1x1
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_output)
    
    # Batch normalization, Flatten, Dense layers
    branch_output = BatchNormalization()(branch1)
    flatten_layer = Flatten()(branch_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])