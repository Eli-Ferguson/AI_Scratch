This is a personal project for me with the aim to deepen my knowledge of neural networks and machine learning
as a whole. This project aims to create something similar to how tensorflow works where you can create
variable size networks and customize them with any mix of activations, node counts, and loss functions.

To run this model simply open the modeRunnerFile.py

The model is instantiated with the Model() class
To set the logging level of the output change the verbose parameter in the
Model class initialization
    verbose=0 : minimal logging
    verbose=1 : simple logging metrics
    verbose=2 : all logging metrics

Next are some steps to setup the model for running

    Step 1: Set Input Shape
        Currently the model only accepts single dimension lists as input such as [1,2,3]
        In this case you would run the command
        Model.setInputDim( numberOfInputs=3 )
    
    Step 2: Set Loss Function
        The current acceptable loss functions are
            'mse' or 'binaryCrossEntropy'
        These are to be passed as strings to the command
        Model.setLossFunc( 'mse' )
    
    Step 3: Add Layers
        For each layer a number of nodes and an Activation function is required
        The current available activation functions are
            'sigmoid' or 'tanh' or 'relu'
            * Note you can set the relu to be a leaky relu by adjusting the slant parameter in the Model.mlFunctions.py file *
        To add a layer use this command
        Model.addLayer( nodes=#, activation='relu' )
            * Note the final layer added before training will be used as output layer *
    
    Step 4: Train
        To train pass input values in a x=[[],[],] format where each inner
            list should be the same len as you set in step 1
        Similarly the y parameter should have shape [[],[],] where the
            inner list should be the same len as the number of nodes in the
            final layer added

        Additionally there are parameters for setting
            batchSize=#
            epochs=#
            lr=#

            * Note these parameters have default values and are not
            required *

        An example training call looks like this for an example with
            input size 1 and output size 1
        
        Model.train( x=[[0.4],[0.6]], y=[[0],[1]], epochs=10 )
    
    Step 5: Predict
        Prediction is not currently supported however you can get a
        prediction on a single input by doing the following

        Model.ForwardPropAllLayers( networkInputs=[[0.4]], train=False )

        * this will return a list of the outputs of the final
        layer of nodes *