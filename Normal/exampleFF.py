


class Model :
    
    def __init__(self) :
        
        self.weights = [ [ -0.1, 0.3, 0.4, 0.6],[ 2, 1 ] ]
        self.inputs = [0.2, 0.8]
    
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
        
        if len(dims) == 1 : return [ dims[0], 1 ]
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
    
    assert divisibilityOfShapesCalc(currentListDims, newShape), f"\nNew Shape :\t{newShape}\nNot Compatible with Current Shape\t{currentList}"
    
    return doReshape(currentList, newShape)

        
test2 = [
    [
        [ [1, 2], [3, 4], [5, 6] ],
        [ [7, 8], [9, 10], [11, 12] ]
    ],
    [
        [ [13, 14], [15, 16], [17, 18] ],
        [ [19, 20], [21, 22], [23, 24] ]
    ],
    [
        [ [1, 2], [3, 4], [5, 6] ],
        [ [7, 8], [9, 10], [11, 12] ]
    ],
    [
        [ [13, 14], [15, 16], [17, 18] ],
        [ [19, 20], [21, 22], [23, 24] ]
    ]
]

reshaped = reshape(test2, [2, 24], verbose=1)

print(reshaped)