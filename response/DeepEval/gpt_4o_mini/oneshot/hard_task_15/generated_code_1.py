import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Add
from keras.models import Model

def dl_model():
    # Input layer for CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    pooled_features = GlobalAveragePooling2D()(input_layer)
    dense1_main = Dense(units=512, activation='relu')(pooled_features)
    dense2_main = Dense(units=3 * 32 * 32, activation='sigmoid')(dense1_main)  # Assuming we want to reshape to match input
    
    # Reshaping to match input layer's shape
    reshaped_weights = Dense(units=32 * 32 * 3, activation='sigmoid')(dense2_main)
    reshaped_weights = keras.layers.Reshape((32, 32, 3))(reshaped_weights)
    
    # Element-wise multiplication with the input feature map
    main_path_output = Multiply()([input_layer, reshaped_weights])

    # Branch path
    branch_path_output = input_layer  # Directly connected to input

    # Combine both paths
    combined_output = Add()([main_path_output, branch_path_output])

    # Feed through two fully connected layers
    combined_flattened = GlobalAveragePooling2D()(combined_output)  # Pool before dense layers for better performance
    final_dense1 = Dense(units=512, activation='relu')(combined_flattened)
    final_output = Dense(units=10, activation='softmax')(final_dense1)  # CIFAR-10 has 10 classes

    # Create model
    model = Model(inputs=input_layer, outputs=final_output)

    return model

# To create the model, simply call dl_model
model = dl_model()
model.summary()  # To display the model architecture