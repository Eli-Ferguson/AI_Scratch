


class Model :
    
    def __init__(self) :
        
        self.weights = [ [ -0.1, 0.3, 0.4, 0.6 ],[ 2, 1 ] ]
        self.inputs = [ 0.2, 0.8 ]
    
model = Model()
  

def Dimensions( inputArray, test=1, verbose=1 ) :
    
    def getDimensions( array ) :
    
        if type( array[ 0 ] ) != list :
            return [ len( array ) ]
            
        ret = [ len( array ) ] + getDimensions( array[ 0 ] )
                    
        return ret 
    
    def TestDimensions( array ) :
    
        if type( array[ 0 ] ) != list :
            return [ array ]
        else:
            ret = [ TestDimensions(ary) for ary in array ]
            
        dims = [ getDimensions( ary ) for ary in ret ]
        
        def assertTruth( val, failedDimensions ) :
            assert val, "All Input Dimensions Must Be Matching Dimensions\t" + str( failedDimensions )
        
        [ assertTruth( truth, dims ) for truth in [ dims[0] == dim for dim in dims ] ]
            
        return dims
    
    
    try :
                        
        if test : TestDimensions( inputArray )
        
        dims = getDimensions( inputArray )
        
        if len(dims) == 1 : return [ 1, dims[0] ]
        else : return dims
            
    except AssertionError as AE:
        if str( AE ).__contains__( "All Input Dimensions Must Be Matching Dimensions" ) :
            if verbose : print(AE)
            return -1
        else :
            raise AssertionError(AE)

def flatten( array ) :
    
    if type( array[ 0 ] ) != list :
        return [ a for a in array ]
    
    ret = []
        
    for ary in array : ret = ret + flatten( ary )
                        
    return ret


def reshape( currentList, newShape, verbose=0 ) :
    
    currentListDims = Dimensions( currentList )
    
    if verbose : print("Dimensions:", currentListDims, newShape)
    
    def calcProductOfList( inputList, prod=1 ) :        
        for val in inputList :prod *= val
        return prod
        
    def divisibilityOfShapesCalc( shape_1, shape_2 ) :
        shape_1_product = calcProductOfList( shape_1 )
        shape_2_product = calcProductOfList( shape_2 )
        
        if verbose == 2 : print("Shape Products:", shape_1_product, shape_2_product)
        
        if shape_1_product == shape_2_product : return 1
        else : return 0
            
    def doReshape( array, shape=None, layer=0 ) :
    
        assert shape is not None, "Shape Must Be Passes an list type\n\tex:[2] or [2,2,2]"
        
        flat = flatten( array )
        
        if( len( shape ) > 1 ) :
            array = doReshape(flat, shape[1:], layer+1)
            shape = shape[0:1]
                
        if shape[0] == 1 and layer == 0: return array
            
        arrayLength = len( array )
        lastItem = len(shape)-1
        divArrayLength = int( arrayLength / shape[ lastItem ] )
        
        if( divArrayLength == 1 ) : return array
        
        new_array = []
            
        for partition in range(0, divArrayLength ) :
            tmp_array = array[ partition * shape[ lastItem ] : shape[ lastItem ] * ( partition+1 ) ]
            new_array.append( tmp_array )
        
        return new_array
    
    assert divisibilityOfShapesCalc(currentListDims, newShape), f"\nNew Shape :\t{newShape}\nNot Compatible with Current Shape\t{Dimensions(currentList)}"
        
    return doReshape(currentList, newShape)


def transpose( matrix ) :
    
    matrixDims = Dimensions( matrix )
    
    assert len( matrixDims ) == 2, f"Input Must Be A 2D Matrix, Input Shape { matrixDims }"
    
    if matrixDims.__contains__( 1 ) :
        matrixDims.reverse()
        return reshape( matrix, matrixDims )
    
    transposed_tuples = list(zip(*matrix))
    transposed = [list(sublist) for sublist in transposed_tuples]
    
    return transposed

def zeros( shape=[], val=0 ) :
    
    def calcProductOfList( inputList, prod=1 ) :        
        for val in inputList :prod *= val
        return prod
    
    length = calcProductOfList( shape )
        
    ret = []
    
    for i in range( 0, length ) :
        ret.append(val)
                
    return reshape( ret, shape )        

def ForwardPropSingleLayer( inputs, weights, activation=None, verbose=0 ) :
    
    inputDims = Dimensions(inputs)
    weightsDims = Dimensions(weights)
    
    if verbose : print(f'InputDims: {inputDims}\tWeightDims: {weightsDims}\tOutputDims: {[ inputDims[1], weightsDims[0] ]}')
    
    assert inputDims[0] == weightsDims[1], f"Non-Compatible Shapes For Matrix Multiplication, Matrix 1 shape {inputDims}, Matrix 2 Shape {weightsDims}"

    def multStep() :
        
        result = zeros( [ inputDims[1], weightsDims[0] ] )
        if verbose : print(f'\tResults: {result}\tInputs: {inputs}\tWeights: {weights}')
        
        dimCheck1_1 = inputDims[0] == 1
        dimCheck1_2 = inputDims[1] == 1
        
        dimCheck2_1 = weightsDims[0] == 1
        dimCheck2_2 = weightsDims[1] == 1
        
        for i in range( 0, inputDims[0] ) :
           if verbose == 2 : print(f'i={i}')
           
           for j in range( 0, weightsDims[0] ) :
               if verbose == 2 : print(f'\tj={j}')
               
               for k in range( 0, inputDims[1] ) :
                    if verbose == 2 : print(f'\t\tk={k}', end='')

               
                    stmt1 = inputs[i][k]
                    stmt2 = weights[j][i] if not dimCheck2_1 else weights[i]
                    
                    if dimCheck2_1 and dimCheck1_2 :
                        result[0] += stmt1 * stmt2
                        if verbose == 2 : print(f'\t', stmt1, stmt2, result[0]) 
                    else :
                        result[j] += stmt1 * stmt2
                        if verbose == 2 : print(f'\t', stmt1, stmt2, result[i]) 
                             
        if verbose : print(f'\tResult Of Mult: {result}')
        return result

    def activationStep() :
        
        if activation is None : return zValues, zValues
        
        aValues = []
        d_aValues = []
        
        for zVal in zValues : aValues.append( activationFunctionsDict.get(activation)[0](zVal) )
        if verbose : print(f'\tPost Activation with : {activation} : {aValues}')

        for zVal in zValues : d_aValues.append( activationFunctionsDict.get(activation)[1](zVal) )
        if verbose : print(f'\tActivation Derivative with : {activation} : {d_aValues}')

        
        return aValues, d_aValues
    
    zValues = multStep()
    
    aValues, d_aValues = activationStep()
    
    return aValues, d_aValues

def ForwardPropAllLayers( networkInputs, allWeights, activations=None, verbose=0 ) :
    
    if activations is None:
        activations = []
        for i in range( 0, len( allWeights ) ) : activations.append( None )
        
    layerOutputs = transpose(networkInputs)
    
    layerForwardOutputs = [layerOutputs]
    layerForwardDerivatives = []
        
    for i in range( 0, len( allWeights ) ) :
        
        layerOutputs, layerDerivatives = ForwardPropSingleLayer( layerOutputs, transpose( allWeights[ i ] ), activations[ i ], verbose )
        layerOutputs = transpose( layerOutputs )
        if verbose : print(f'\nlayerOutputs: {layerOutputs}\n')
        
        layerDerivatives = transpose( layerDerivatives )
        if verbose == 2 : print(f'LayerInformation: {list( zip( layerOutputs, layerDerivatives ) )}\n')
        
        layerForwardOutputs.append( layerOutputs )
        layerForwardDerivatives.append( layerDerivatives )

    
    return layerOutputs, layerForwardOutputs, layerForwardDerivatives


inputs = [ 1.1, 0.4 ]
weights = [ 
            [ [ 0.3, -0.3 ], [ -0.4, 0.6 ] ],
            [ [0.9], [0.3] ]
        ]

def sigmoid( val, e=2.718281828459045 ) :
    return 1 / ( 1 + e**(-val) )

def dSigmoid( val ) :
    return sigmoid( val ) * ( 1 - sigmoid( val ) )

def tanh( val, e=2.718281828459045 ) :
    return ( e**val - e**( -val ) ) / ( e**val + e**( -val ) )

def dTanh( val ) :
    return ( 1 - tanh( val )**2 )

activationFunctionsDict = {
    'sigmoid':[ sigmoid, dSigmoid ],
    'tanh': [ tanh, dTanh ]
}

def meanSquaredError( true, pred, verbose=0 ) :
    
    if type( true ) == int or type( true ) == float : true = [ true ]
    if type( pred ) == int or type( pred ) == float : pred = [ pred ]
    
    assert len( true ) == len( pred ), f"Batch Size of True Values and Predicted Values must be equal\nTrue Batch Length: {len( true )}\tPred Batch Length: {len( pred )}"
    
    summedError = 0
    for singleTrue, singlePred in list( zip( true, pred ) ) :
        summedError += 0.5 * ( singleTrue - singlePred )**2
    
    if verbose : print(f'Batch Of Length: { len(true) } Has meanSquaredError Error = {summedError}')
    return summedError
    
def dMeanSquaredError( true, pred, verbose=0 ) :
    if type( true ) == int or type( true ) == float : true = [ true ]
    if type( pred ) == int or type( pred ) == float : pred = [ pred ]
    
    assert len( true ) == len( pred ), f"Batch Size of True Values and Predicted Values must be equal\nTrue Batch Length: {len( true )}\tPred Batch Length: {len( pred )}"
    
    assert type( true[ 0 ] ) != list and type( pred[ 0 ] ) != list, f"Derivative of Mean Squared Error Only Supports Single Node Output\n"
    
    summedError = 0
    for singleTrue, singlePred in list( zip( true, pred ) ) :
        summedError += -( singleTrue - singlePred )
    
    if verbose : print(f'Batch Of Length: { len(true) } Has A dMeanSquaredError = {summedError}')
    return [ summedError ]

lossFunctionsDict = {
    'mse' : [ meanSquaredError, dMeanSquaredError ]
}

def BackPropLayers( true, outputs, derivatives, currentWeights, loss, verbose=0 ) :
    
    outputs.reverse()
    derivatives.reverse()
    weights.reverse()
    
    predictions = outputs[0]
    
    weights.insert( 0, [ zeros( [ len( predictions ), 1 ], 1 ) ] )
    # derivatives.insert( 0, lossFunctionsDict.get( loss )[ 1 ]( true, predictions, verbose) )
    
    if verbose == 2 : print( f'Predictions: {predictions}\nOutputs: {outputs}\n\nDerivatives: {derivatives}\nCurrentWeights: {currentWeights}\n')
    
    dLoss = [ lossFunctionsDict.get( loss )[ 1 ]( true, predictions, verbose )[0] * derivatives[0][0]]
    batchLoss = lossFunctionsDict.get( loss )[0]( true, predictions, verbose )
    
    dL_da = [dLoss]
    
    for layer in range( 1, len( outputs ) - 1 ) :
        
        dL_da.append( zeros( [ 1, len( weights[ layer ] ) ] ) )        
        
        if verbose == 2: print(f'dL/da: {dL_da}\nLayer: {layer}')
                
        # print(len( weights[ layer ] ), weights[ layer ] )
        
        for node in range( 0, len( weights[ layer ] ) ) :
            
            if verbose == 2 : print(f'\n\tCurrent Layer|Node : {layer}|{node}')
            
            #Get Weights out
            if verbose == 2 : print(f'\t\tWeights @ Current Layer|Node {layer}|{node} : { weights[ layer ][ node ] }')
            
            #Get Prev Layer dL/da
            if verbose == 2 : print(f'\t\tdL/da @ Prev Layer {layer-1} : { dL_da[ layer - 1 ] }')
            
            def crossProduct( l1, l2 ) :
                
                ret = 0
                                
                for i in l1 :
                    for j in l2 :
                        ret += i * j
                
                return ret
            
            cp = crossProduct( weights[ layer ][ node ], dL_da[ layer - 1 ] )
                        
            dL_da[ layer ][ node ] = cp * derivatives[ layer ][ node ][0]
    
    if verbose == 2 : print(dL_da)
    return dL_da, batchLoss
        

def updateWeights( weights, outputs, partials, lr, verbose) :
    
    weights.pop(0)
    outputs.pop(0)
    if verbose == 2 : print(f'outputs: {outputs}\n\tPartials: {partials}\n\tLearning Rate: {lr}')

    gradients = []

    for layer in range( 0, len( weights ) ) :
        
        if verbose == 2 : print(f'\n\t\tPartial @ layer {layer} : {partials[layer]}')
        if verbose == 2 : print(f'\t\tOutput @ layer {layer} : {outputs[layer]}')

        # for node in range( 0, len( weights[ layer ] ) ) :
        #     print(f'\t\tOutput @ layer {layer}|{node} : {outputs[layer][node]}')
        
        def allCombinations( l1, l2 ) :
            
            ret = []
                                
            for i in l2 :
                for j in l1 :
                    ret.append( i[0] * j )
            
            return ret
        
        gradients.append( allCombinations( partials[layer], outputs[layer] ) )
        
    for layer in range( 0, len( weights ) ) :
        
        if verbose == 2 : print(f'Current Weights: {weights[layer]}\nWeight Gradients: {gradients[layer]}')
        
        idx = 0
        for nodeOuts in range( 0, len( weights[layer] ) ) :
            if verbose == 2 : print(f'\tNode Out Weight: {weights[layer][nodeOuts]}')
            
            for singleWeight in range( 0, len( weights[layer][nodeOuts] ) ) :
                if verbose == 2 : print(f'\t\tSingle Weight: {weights[layer][nodeOuts][singleWeight]}\tidx: {idx}')
                
                if verbose == 2 : print(f'\t\tNew Weight = {weights[layer][nodeOuts][singleWeight] - lr*gradients[layer][idx]}')

                weights[layer][nodeOuts][singleWeight] = weights[layer][nodeOuts][singleWeight] - lr*gradients[layer][idx]
                
                idx+=1
        
        if verbose == 2 : print(f'\nNew Layer Weights: {weights[layer]}\n')

    if verbose == 1 : print(f"Updated Weights: {weights}")
    weights.reverse()
    return weights
                
           
def runNN( inputs, expected, weights, lr=0.1, epochs=1 , verbose=0) :
    
    pred = []
    lossHistory = []
    
    for i in range( 0, epochs ) :
        if verbose: print(f'Part 1.{i}: ForwardPropagation')
        FP_prediction, FP_layerOutputs, FP_layerDerivatives = ForwardPropAllLayers( inputs, weights, ['tanh', 'sigmoid'], verbose )
        pred.append(FP_prediction)
        
        if verbose: print(f'\n\nPart 2.{i}: BackPropagation')
        partial_derivatives, batchLoss = BackPropLayers( true, FP_layerOutputs, FP_layerDerivatives, weights, 'mse', verbose)
        lossHistory.append(batchLoss)
        
        if verbose: print(f'\n\nPart 3.{i}: Update Weights')
        weights = updateWeights( weights, FP_layerOutputs, partial_derivatives, lr=lr, verbose=verbose)
    
    if verbose == 1:
        print(f'\nFirst Predictions: {pred[0]}\nFirst Loss: {lossHistory[0]}')
    
    if verbose == 2: print(f'\nAll Predictions: {pred}\nAll Loss: {lossHistory}')
    
    return pred[-1], lossHistory[-1]

inputs = [ 1.1, 0.4 ]
weights = [ 
            [ [ 0.3, -0.3 ], [ -0.4, 0.6 ] ],
            [ [0.9], [0.3] ]
        ]
true = [0.01]

finalPred, finalLoss  = runNN( inputs, true, weights, lr=0.9, epochs=1000, verbose=0)

print(f'\nFinal Prediction: {finalPred}\nFinal Loss: {finalLoss}\n')
