from Model.Model import *

M = Model( verbose=0 )

M.setInputDim( numberOfInputs=1 )
M.setLossFunc( 'binaryCrossEntropy' )

M.addLayer( 20, 'sigmoid' )
M.addLayer( 10, 'sigmoid' )
M.addLayer( 1, 'sigmoid' )

x = [ [xi] for xi in generateListOfNums(200) ]
y = [ round(xi[0]) for xi in x ]
print(f'\n\nx:{x}\n\ny:{y}\n\n')

M.train( x, y, batchSize=5, epochs=100, lr=0.9 )

print('\n\n')
for xi in x[-5:] :
    print(f'Prediction on {xi}:{ [ round( pred, 2 ) for pred in M.ForwardPropAllLayers( [ xi ] )[0] ]}' )