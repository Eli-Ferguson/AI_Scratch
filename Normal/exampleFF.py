


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
        
        def assertTruth(val, failedDimensions) :
            assert val, "All Input Dimensions Must Be Matching Dimensions\t" + str(failedDimensions)
        
        [ assertTruth(truth, dims) for truth in [dims[0] == dim for dim in dims] ]
            
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
            
# test = [
#     [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]],
#     [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]],
#     [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]],
#     [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]],
#     [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]],
#     [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]],
#     [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]
#     ]

# print( Dimensions(test, 0) )


def reshape( currentList, newShape, verbose=1 ) :
    
    currentListDims = Dimensions( currentList )
    
    if verbose : print("Dimensions:", currentListDims, newShape)
    
    def calcProductOfList( inputList, prod=1 ) :        
        for val in inputList :prod *= val
        return prod
        
    def divisibilityOfShapesCalc( shape_1, shape_2 ) :
        shape_1_product = calcProductOfList( shape_1 )
        shape_2_product = calcProductOfList( shape_2 )
                
        mod1 = shape_1_product / shape_2_product % 1
        mod2 = shape_2_product / shape_1_product % 1
                
        if not mod1 or not mod2:return 1
        else:return 0
        
    assert divisibilityOfShapesCalc(currentListDims, newShape), f"\nNew Shape :\t{newShape}\nNot Compatible with Current Shape\t{currentList}"
    
    
list = [1, 2, 3, 4]

shape = [2, 2]

reshaped = [[1,2],[3,4]]

reshape(list, shape)

# first, i create some data
l = [ i for i in range(10) ]
# now I reshape in to slices of 4 items
x = [ l[x:x+4] for x in range(0, len(l), 4) ] 
print(x)
    