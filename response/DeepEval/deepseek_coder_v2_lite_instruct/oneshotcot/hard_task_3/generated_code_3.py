import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, Dropout, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Define the main pathway
    def pathway(input_tensor):
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1x1)
        dropout = Dropout(0.5)(conv3x3)  # Feature selection to mitigate overfitting
        return dropout
    
    # Apply the pathway to each group and concatenate the results
    pathways = [pathway(split[i]) for i in range(3)]
    main_pathway = Concatenate()(pathways)
    
    # Parallel branch pathway
    branch = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Add the main pathway and the branch pathway
    added = Add()([main_pathway, branch])
    
    # Flatten the output and pass it through a fully connected layer
    flatten = Flatten()(added)
    dense = Dense(units=10, activation='softmax')(flatten)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=dense)
    
    return model

# Build the model
model = dl_model()
model.summary()