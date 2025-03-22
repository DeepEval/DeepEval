# Import necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Define the main path of the model
def main_path(input_shape):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    return Model(inputs, x)

# Define the branch path of the model
def branch_path(input_shape):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    return Model(inputs, x)

# Combine the main and branch paths
def combined_path(main_path, branch_path):
    x = main_path.output
    branch_x = branch_path.output
    
    # Use an addition operation to combine the two paths
    x = Add()([x, branch_x])
    
    # Flatten the combined output into a one-dimensional vector
    x = Flatten()(x)
    
    # Project the features onto a probability distribution across 10 classes
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Dense(10, activation='softmax')(x)
    
    return Model(main_path.inputs, x)

# Define the deep learning model
def dl_model():
    input_shape = (32, 32, 3)
    
    main_path_model = main_path(input_shape)
    branch_path_model = branch_path(input_shape)
    
    combined_model = combined_path(main_path_model, branch_path_model)
    
    # Compile the model
    combined_model.compile(optimizer=Adam(lr=0.001),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
    
    return combined_model