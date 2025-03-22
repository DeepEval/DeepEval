import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Split the input into three groups along the channel dimension
    split = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Define the paths for each group
    path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split[0])
    path1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(path1)
    path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(path1)
    
    path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split[1])
    path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(path2)
    path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(path2)
    
    path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split[2])
    path3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(path3)
    path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(path3)
    
    # Step 3: Combine the outputs from the three groups using addition
    combined_path = Add()([path1, path2, path3])
    
    # Step 4: Add the combined path back to the original input
    fused_path = Add()([combined_path, input_layer])
    
    # Step 5: Flatten the combined features
    flatten_layer = Flatten()(fused_path)
    
    # Step 6: Add a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Step 7: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model