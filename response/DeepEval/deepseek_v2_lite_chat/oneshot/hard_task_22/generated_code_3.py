import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Activation
from keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel
    x = Lambda(lambda tensors: keras.backend.split(tensors, 3, axis=-1))(inputs)
    
    # Feature extraction paths
    path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x[0])
    path2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x[1])
    path3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(x[2])
    
    # Concatenate the outputs of the three paths
    concat = Concatenate(axis=-1)([path1, path2, path3])
    
    # Main path output
    main_output = Flatten()(concat)
    
    # Branch path
    branch_input = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(inputs)
    branch_output = Flatten()(branch_input)
    
    # Fusion of main and branch paths
    fused_output = keras.layers.Add()([main_output, branch_output])
    
    # Fully connected layers
    fc1 = Dense(units=256, activation='relu')(fused_output)
    fc2 = Dense(units=128, activation='relu')(fc1)
    output = Dense(units=10, activation='softmax')(fc2)
    
    # Model construction
    model = Model(inputs=inputs, outputs=output)
    
    return model

# Build the model
model = dl_model()
model.summary()