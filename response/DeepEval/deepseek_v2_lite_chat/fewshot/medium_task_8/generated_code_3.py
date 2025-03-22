import keras
from keras.layers import Input, Lambda, Conv2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Main path
    main_path = split_layer[0]
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(main_path)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(main_path)
    
    # Branch path
    branch_path = split_layer[1]
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(branch_path)
    
    # Concatenate outputs from main and branch paths
    concat_layer = Add()([conv1, conv2, conv3])
    
    # Flatten and pass through fully connected layer
    flatten_layer = Flatten()(concat_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()

# Print model summary
model.summary()