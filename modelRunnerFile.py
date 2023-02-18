from Model.Model import *

M = Model( verbose=0 )

M.setInputDim( numberOfInputs=2 )
M.setLossFunc( 'mse' )

M.addLayer( 20, 'tanh' )
M.addLayer( 20, 'tanh' )
M.addLayer( 10, 'tanh' )
M.addLayer( 10, 'tanh' )
M.addLayer( 1, 'sigmoid' )

M.train( [ [ 1.1, 0.4 ] ], [ 0.5 ], epochs=100, lr=0.9 )