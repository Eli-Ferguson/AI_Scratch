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
    
    if verbose == 2 : print( f'True Values:{true}\nPred Values:{pred}' )
            
    assert Dimensions(true) == Dimensions(pred), f"Dims of True Values and Predicted Values must be equal\nTrue Dims: {Dimensions(true)}\tPred Dims: {Dimensions(pred)}"
    
    summedError = []
    for singleTrue, singlePred in list( zip( true, pred ) ) :
        try : summedError.append( 0.5 * ( singleTrue[0] - singlePred[0] )**2 )
        except : summedError.append( 0.5 * ( singleTrue - singlePred )**2 )
    
    if verbose == 2 : print(f'Layer With { len(true) } Output Nodes Has meanSquaredError Error = {summedError}')
    return summedError
    
def dMeanSquaredError( true, pred, verbose=0 ) :
    
    if verbose == 2 : print( f'True Values:{true}\nPred Values:{pred}' )
                
    assert Dimensions(true) == Dimensions(pred), f"Dims of True Values and Predicted Values must be equal\nTrue Dims: {Dimensions(true)}\tPred Dims: {Dimensions(pred)}"
        
    summedError = []
    for singleTrue, singlePred in list( zip( true, pred ) ) :
        try : summedError.append( singlePred[0] - singleTrue[0] )
        except : summedError.append( singlePred - singleTrue )
    
    if verbose == 2 : print(f'Layer With { len(true) } Output Nodes Has A dMeanSquaredError = {summedError}')
    return summedError

# dMeanSquaredError( [[1], [1]], [[0.9094020863185921], [0.9280105706677314]], verbose=2 )
# dMeanSquaredError( [[1], [1]], [[0.9094020863185921], [0.9280105706677314]], verbose=2 )

lossFunctionsDict = {
    'mse' : [ meanSquaredError, dMeanSquaredError ]
}

def isPrime( num:int ) :
    
    return len( [ x for x in range( 2, num ) if num % x == 0 ] ) == 0

def bsd_rand(seed): 
   def rand(): 
      rand.seed = (1103515245*rand.seed + 12345) & 0x7fffffff 
      return rand.seed 
   rand.seed = seed 
   return rand

def generateListOfNums( length:int, mod:int=99999, divisor:int=100000, randSeed:int=1 ) :
    
    r = bsd_rand( randSeed )
    
    ret = []
        
    while len(ret) < length :
        tmp = r() % mod
        if isPrime( tmp ) and tmp not in ret : ret.append( tmp / divisor )
    
    return ret

def generateWeights( inCount:int, layerCount:int, randSeed:int ) :
    return [ generateListOfNums( length=layerCount, randSeed=randSeed ) for _ in range( inCount ) ]

def crossProduct( l1, l2 ) :
                    
    ret = 0
                    
    for i in l1 :
        for j in l2 :
            ret += i * j
    
    return ret


