from Model.Model import *

M = Model( verbose=0 )

M.setInputDim( numberOfInputs=1 )
M.setLossFunc( 'mse' )
M.setLossFunc( 'binaryCrossEntropy' )

# M.addLayer( 10, 'tanh' )
M.addLayer( 1, 'relu' )
M.addLayer( 1, 'sigmoid' )

x = [ [xi] for xi in generateListOfNums(20) ]
y = [ round(xi[0]) for xi in x ]
print(f'\n\nx:{x}\n\ny:{y}\n\n')


x = [ [0.1] ]
y = [ [ 0 ] ]

M.train( x, y, batchSize=1, epochs=100, lr=0.9 )

print('\n\n')
for xi in x[-5:] :
    print(f'Prediction on {xi}:{ [ pred for pred in M.ForwardPropAllLayers( [ xi ] )[0] ]}' )
    # print(f'Prediction on {xi}:{ [ round( pred ) for pred in M.ForwardPropAllLayers( [ xi ] )[0] ]}' )