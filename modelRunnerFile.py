from Model.Model import *

M = Model( verbose=0 )

M.setInputDim( numberOfInputs=1 )
M.setLossFunc( 'binaryCrossEntropy' )

# M.addLayer( 20, 'sigmoid' )
M.addLayer( 10, 'sigmoid' )
M.addLayer( 1, 'sigmoid' )

# Generate a list of random numbers between 0 and 1
# Round these numbers to get classes, either 0 or 1
x = [ [xi] for xi in generateListOfNums(1000) ]
y = [ round(xi[0]) for xi in x ]
print(f'\n\nx:{x}\n\ny:{y}\n\n')

# Train Model
M.train( x, y, batchSize=1, epochs=10, lr=0.1 )

M.verbose = 0

# Show Predictions
print('\n\n')
for xi in x[-5:] :
    print(f'Prediction on {xi}:{ [ round( pred, 2 ) for pred in M.ForwardPropAllLayers( [ xi ] )[0] ]}' )