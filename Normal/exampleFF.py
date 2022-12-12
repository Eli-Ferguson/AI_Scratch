


class Model :
    
    def __init__(self) :
        
        self.weights = [ [ -0.1, 0.3, 0.4, 0.6 ],[ 2, 1 ] ]
        self.inputs = [ 0.2, 0.8 ]
    
model = Model()
  

def Dimensions( inputArray, verbose=1 ) :
    
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
                        
        TestDimensions( inputArray )
        
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

def zeros( shape=[] ) :
    
    def calcProductOfList( inputList, prod=1 ) :        
        for val in inputList :prod *= val
        return prod
    
    length = calcProductOfList( shape )
        
    ret = []
    
    for i in range( 0, length ) :
        ret.append(0)
                
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
        
        if activation is None : return zValues
        
        aValues = []
        
        for zVal in zValues : aValues.append( activation(zVal) )
        
        if verbose : print(f'\tPost Activation with : {activation.__name__} : {aValues}')
        return aValues
    
    zValues = multStep()
    
    aValues = activationStep()
    
    return aValues

def ForwardPropAllLayers( networkInputs, allWeights, activations=None, verbose=0 ) :
    
    if activations is None:
        activations = []
        for i in range( 0, len( allWeights ) ) : activations.append( None )
        
    layerInputs = transpose(networkInputs)
    
    for i in range( 0, len( allWeights ) ) :
        
        layerInputs = transpose( ForwardPropSingleLayer( layerInputs, transpose( allWeights[ i ] ), activations[ i ], verbose ) )
        if verbose : print(f'\nlayerInputs: {layerInputs}\n')
    
    return layerInputs

# inputs = [ 0.2, 0.8, 0.5 ]
# weights = [ [ -0.1, 0.3 ], [ 0.4, 0.6 ], [0.5, 0.5] ]

inputs = [ 1.1, 0.4 ]
weights = [ 
            [ [ 0.3, -0.3 ], [ -0.4, 0.6 ] ],
            [ [0.9], [0.3] ]
        ]

# inputs = transpose(inputs)
# weights = transpose(weights[0])

print(inputs)
print(weights, '\n\n')

def sigmoid( val, e=2.718281828459045 ) :
    return 1 / ( 1 + e**(-val) )

def tanh( val, e=2.718281828459045 ) :
    return ( e**val - e**( -val ) ) / ( e**val + e**( -val ) )

# print( ForwardPropSingleLayer( inputs, weights, verbose=2 ) )
print( ForwardPropAllLayers( inputs, weights, [tanh, sigmoid], verbose=0 ) )