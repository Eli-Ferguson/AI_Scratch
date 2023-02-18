try :
    from mlFunctions import *
except :
    from Model.mlFunctions import *


class Model :
    def __init__( self, verbose:int=2 ) :
        
        self.inputDim = []
        self.weights = []
        self.activations = []
        
        self.iterationCount = 0
        self.layersCount = 0
        self.activationDict = activationFunctionsDict
        self.lossFuncDict = lossFunctionsDict
        
        self.currentLayer = 0
        self.verbose = verbose
        
        self.history = []
        
    def train( self, x:list, y:list, epochs:int=5, lr:float=0.9 ) :
        
        assert len(x) == len(y), f'X and Y must be of same length\n\tx:{len(x)}\ty:{len(y)}'
        assert lr > 0, f'Learning Rate Must Be Greater Than 0\n\tlr={lr}'
        
        self.lr = lr
                
        try :
            x = [ transpose( xi ) for xi in x ]
        except :
            x = [ [ xi ] for xi in x ]
        
        try :
            y = [ transpose( yi ) for yi in y ]
        except :
            y = [ [ yi ] for yi in y ]
                    
        data = list( zip( x, y ) )
        
        if self.verbose : print(f'data:{data}')
        if self.verbose : print(f'Weights = {self.weights}')

        for e in range( epochs ) :
            
            print(f'\n\nEPOCH #{e+1}\n')
            
            for xi, yi in data :
                                
                xi = xi if type(xi[0]) == list else [ xi ]
                yi = yi if type(yi[0]) == list else [ yi ]
                                
                self.ForwardPropAllLayers( xi )
                
                self.BackPropLayers( yi )
                
                self.updateWeights()
                
                # print(f'Weights = {self.weights}')
                print(f'\tPrediction:{self.history[-1][0]["layerOutputs"][0]}')
                print(f'\tLoss:{self.iterationLoss}')
        
    def ForwardPropAllLayers( self,  networkInputs ) :
        
        d = Dimensions( networkInputs )
                
        assert len(d) == 2, f'\nInput contains too many dimensions: {d}\n'
        assert d[0] == 1 or d[1] == 1, f'\nOne Dimension must be 1: {d}\n'
            
        layerOutputs = networkInputs if Dimensions( networkInputs )[1] == 1 else transpose(networkInputs)
        
        layerForwardOutputs = [layerOutputs]
        layerForwardDerivatives = []
                    
        for i in range( 0, self.layersCount ) :
            
            layerOutputs, layerDerivatives = self.ForwardPropSingleLayer( layerOutputs, i )
            if self.verbose : print(f'\nlayerOutputs: {layerOutputs}\n')
            
            if self.verbose == 2 : print(f'LayerInformation: {list( zip( layerOutputs, layerDerivatives ) )}\n')
            
            layerForwardOutputs.append( layerOutputs )
            layerForwardDerivatives.append( layerDerivatives )
            
            # self.goDownALayer()

        self.history.append([])
        self.history[ self.iterationCount ].append( {'layerOutputs':layerOutputs, 'layerForwardOutputs':layerForwardOutputs, 'layerForwardDerivatives':layerForwardDerivatives} )
        
        self.iterationCount+=1

    def ForwardPropSingleLayer(self, inputsToLayer:list, currentLayer:int ) :
        
        weights = self.weights[currentLayer].copy()
        # weights = transpose( self.weights[currentLayer].copy() ) if currentLayer == 0 else self.weights[currentLayer].copy()
        
        inputDims = Dimensions( inputsToLayer )        
        weightsDims = Dimensions( weights )
                
        if self.verbose : print(f'InputDims: {inputDims}\tWeightDims: {weightsDims}\tOutputDims: {[ inputDims[1], weightsDims[0] ]}')
        
        assert inputDims[0] == weightsDims[1], f"Non-Compatible Shapes For Matrix Multiplication, Matrix 1 shape {inputDims}, Matrix 2 Shape {weightsDims}"

        def multStep() :
            
            result = zeros( [ inputDims[1], weightsDims[0] ] )
            if self.verbose : print(f'\tResults: {result}\tInputs: {inputsToLayer}\tWeights: {weights}')
            
            for inNode in range( 0, inputDims[0] ) :
                if self.verbose == 2 : print(f'inNode={inNode}')
                
                for layerNode in range( 0, weightsDims[0] ) :
                    if self.verbose == 2 : print(f'\tlayerNode={layerNode}')
                    
                    for inNodeM in range( 0, inputDims[1] ) :
                            if self.verbose == 2 : print(f'\t\tinNodeM={inNodeM}', end='')
                    
                            stmt1 = inputsToLayer[inNode][inNodeM]
                            stmt2 = weights[layerNode][inNode]
                            
                            result[layerNode] += stmt1 * stmt2
                            
                            if self.verbose == 2 : print(f'\t', stmt1, stmt2, result[layerNode]) 
                                
            if self.verbose : print(f'\tResult Of Mult: {result}')
            return result

        def activationStep() :
            
            if self.activations is None : return zValues, zValues
            
            aValues = transpose( [ self.activations[ currentLayer ][0](zVal) for zVal in zValues ] )
            if self.verbose : print(f'\tPost Activation with : {self.activations[ currentLayer ][0].__name__} : {aValues}')

            d_aValues = transpose( [ self.activations[ currentLayer ][1](zVal) for zVal in zValues ] )
            if self.verbose : print(f'\tActivation Derivative with : {self.activations[ currentLayer ][1].__name__} : {d_aValues}')
                        
            aValues = aValues if type(aValues[-1] ) == list else [aValues]
            d_aValues = d_aValues if type(d_aValues[-1] ) == list else [d_aValues]
                        
            return aValues, d_aValues
        
        zValues = multStep()
        
        aValues, d_aValues = activationStep()
        
        return aValues, d_aValues
    
    def BackPropLayers( self, trues ) :
    
        outputs = self.history[-1][0]['layerForwardOutputs'].copy()
        derivatives = self.history[-1][0]['layerForwardDerivatives'].copy()
        weights = self.weights.copy()        
    
        outputs.reverse()        
        derivatives.reverse()
        weights.reverse()
                                        
        predictions = outputs[0]
                
        if self.verbose == 2 : print( f'Predictions: {predictions}\nOutputs: {outputs}\n\nDerivatives: {derivatives}\nCurrentWeights: {self.weights}\n')
                
        losses = list( zip( [ self.lossFunc[1]( true, pred, self.verbose ) for true, pred in list( zip( trues, predictions ) ) ], derivatives[0] ) )
        
        if self.verbose == 2 : print(f'losses:{losses}')
        
        dLoss = [ lossAtNode[0] * DerivativeAtNode[0] for lossAtNode, DerivativeAtNode in losses]
        
        self.iterationLoss = self.lossFunc[0]( trues, predictions, self.verbose )
        
        dL_da = [dLoss]
                
        for layer in range( len( weights ) - 1 ) :
            
            dL_da.append( zeros( [ 1, len( weights[layer+1] ) ] ) )
            
            for node in range( len( weights[layer] ) ) :                
                
                for connectingWeight in range( len( weights[layer][node] ) ) :
                    
                    if self.verbose == 2 : print(f'\n\tCurrent Layer|Node|Connection : {layer}|{node}|{connectingWeight}')

                    #Get Weights out
                    if self.verbose == 2 : print(f'\t\tWeights @ Connection : { weights[ layer ][ node ][ connectingWeight ] }')
                    
                    #Get Prev Layer dL/da
                    if self.verbose == 2 : print(f'\t\tdL/da @ Prev Layer {layer-1} : { dL_da[ layer ] }')
                
                    cp = crossProduct( [ weights[ layer ][ node ][ connectingWeight ] ], dL_da[ layer ] )                    
                    
                    dL_da[ layer+1 ][ connectingWeight ] = cp * derivatives[ layer+1 ][ connectingWeight ][0]
                    
        
        if self.verbose : print(f'\ndl_da {dL_da}\n')
        self.dL_da = dL_da
    
    def updateWeights( self ) :
                
        outputs = self.history[-1][0]['layerForwardOutputs'][:-1].copy()
        outputs.reverse()
        
        weights = self.weights.copy()
        weights.reverse()
    
        # weights.pop(0)
        if self.verbose == 2 : print(f'\toutputs: {outputs}\n\tPartials: {self.dL_da}\n\tLearning Rate: {self.lr}')
        

        gradients = []

        for layer in range( 0, len( weights ) ) :
            
            if self.verbose == 2 : print(f'\n\t\tPartial @ layer {layer} : {self.dL_da[layer]}')
            if self.verbose == 2 : print(f'\t\tOutput @ layer {layer} : {outputs[layer]}')

            # for node in range( 0, len( weights[ layer ] ) ) :
                # print(f'\t\tOutput @ layer {layer}|{node} : {outputs[layer][node]}')
            
            def allCombinations( l1, l2 ) :
                
                ret = []
                                    
                for i in l2 :
                    for j in l1 :
                        ret.append( i[0] * j )
                
                return ret
            
            gradients.append( allCombinations( self.dL_da[layer], outputs[layer] ) )

        if self.verbose : print(f'\nGradients:{gradients}\n')
        
        for layer in range( 0, len( weights ) ) :
            
            if self.verbose == 2 : print(f'weights[{layer}] = {weights[layer]}\nWeight Gradients: {gradients[layer]}\n')
            
            idx = 0
            
            for node in range( 0, len( weights[layer] ) ) :
                if self.verbose == 2 : print(f'\tnode[{node}] = {weights[layer][node]}\n')
                
                for connectionWeight in range( 0, len( weights[layer][node] ) ) :
                    # if self.verbose == 2 : print(f'\t\tconnection[{connectionWeight}] = {weights[layer][node][connectionWeight]}')
                    
                    # if self.verbose == 2 : print(f'\t\t\tGradient[{layer}][{idx}] = {gradients[layer][idx]}\n')
                    
                    if self.verbose == 2 : print(f'\t\tidx:{idx}')
                    if self.verbose == 2 : print(f'\t\t\tSingle Weight: {weights[layer][node][connectionWeight]}')
                    if self.verbose == 2 : print(f'\t\t\tSingle Gradient: {gradients[layer][idx]}')
                    if self.verbose == 2 : print(f'\t\t\tNew Weight = {weights[layer][node][connectionWeight] - self.lr*gradients[layer][idx]}')
                    
                    weights[layer][node][connectionWeight] = weights[layer][node][connectionWeight] - self.lr*gradients[layer][idx]
                    idx+=1

        if self.verbose : print(f"Updated Weights: {weights}")
        weights.reverse()
        self.weights = weights
    
    def addLayer( self, nodes:int, activation:str ) :
                
        if self.layersCount == 0 :
            # w = generateWeights( self.inputDim[0], nodes, self.layersCount+1 )
            w = transpose( generateWeights( self.inputDim[0], nodes, self.layersCount+1 ) )
        else :
            # w = generateWeights( len( self.weights[-1] ), nodes, self.layersCount+1 )
            w = transpose( generateWeights( len( self.weights[-1] ), nodes, self.layersCount+1 ) )
                
        if Dimensions(w)[0] == 1 : w = [w]
               
        self.weights.append(
            # generateWeights( len( self.weights[-1] ), nodes, self.layersCount+1 )
            # transpose( generateWeights( len( self.weights[-1] ), nodes, self.layersCount+1 ) )
            w
        )
        
        self.activations.append( self.activationDict[ activation ] )
        
        self.layersCount+=1
    
    def calculateLossOfBatch( self, trues, batchSize ) :
        
        batchHistory = self.history[-batchSize:]
        
        losses = []
        
        for i in range( len( batchHistory ) ) :
            if self.verbose : print( f'Iteration {i+1} output : {batchHistory[i][0]["layerOutputs"]}' )
            losses.append( self.lossFunc[1]( trues[i], batchHistory[i][0]['layerOutputs'], verbose=self.verbose ) )
        
        losses = list( zip( *losses ) )
        if self.verbose == 2 : print( f'Losses Zipped : {losses}' )
                
        batchLoss = []
        
        for i in range( len( losses ) ) :
            batchLoss.append(0)
            for j in range( len( losses[0] ) ) :
                batchLoss[ i ] += losses[i][j]
                       
        if self.verbose : print(f'Batch Loss : {batchLoss}')
        
        self.batchLoss = batchLoss
    
    def goDownALayer( self ) : self.currentLayer+=1
    
    def goUpALayer( self ) : self.currentLayer-=1
    
    def setInputDim( self, numberOfInputs:int ) :
        self.inputDim = [ numberOfInputs, 1 ]
        
    def setLossFunc( self, lossFunc:str ) :
        self.lossFunc = self.lossFuncDict[ lossFunc ]
        
    def setLearningRate( self, lr:float ) :
        self.lr = lr
    

