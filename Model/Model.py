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
        
    def train( self, x:list, y:list, batchSize:int=None, epochs:int=5, lr:float=0.9 ) :
        
        assert len(x) == len(y), f'X and Y must be of same length\n\tx:{len(x)}\ty:{len(y)}'
        assert lr > 0, f'Learning Rate Must Be Greater Than 0\n\tlr={lr}'
        
        self.lr = lr
        
        if batchSize == None : batchSize=len(x)
        self.batchSize = batchSize
        
        
        self.numOfBatches = ceil( len(x) / batchSize ) 
        
        print(f'BatchSize:{batchSize}\tNum Of Batches:{self.numOfBatches}')
                
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
            
            print(f'EPOCH #{e+1}')
            
            for batch in range( self.numOfBatches ) :
                
                
                batchX = data[ batch*self.batchSize : (batch+1) * self.batchSize]
                if self.verbose : print( f'\tBatchX:{batchX}' )
            
                for xi, yi in batchX :
                                    
                    xi = xi if type(xi[0]) == list else [ xi ]
                    yi = yi if type(yi[0]) == list else [ yi ]
                    
                    if self.verbose : print('\nForward Step\n')
                    self.ForwardPropAllLayers( xi, train=True )
                
                # print( list( zip( *batchX ) ) )
                # self.calculateLossOfBatch( list( zip( *batchX ) )[1] )
                
                if self.verbose : print('\nBackward Step\n')
                
                trues = list( zip( *batchX ) )[1]
                trues = [ true if type(true[0]) == list else [ true ] for true in trues ]
                
                self.BackPropLayers( trues )
                
                if self.verbose : print('\nUpdate Step\n')
                self.updateWeights()
                                
                # print(f'Weights = {self.weights}')
                if self.verbose == 1 : print(f'\tActual:{yi:.5f}')
                if self.verbose == 1 : print(f'\tPrediction:{[ history[0]["layerOutputs"][:] for history in self.history[-batchSize:] ]}')
                if self.verbose == 1 : print(f'\tLoss @ Batch:{batch} | {self.iterationLoss:.5f}')
                
                if self.verbose == 0 :
                    print(f'\tLoss @ Batch:{batch} | {self.iterationLoss:.5f}')
                        
    def ForwardPropAllLayers( self,  networkInputs, train:bool=False ) :
        
        d = Dimensions( networkInputs )
                
        assert len(d) == 2, f'\nInput contains too many dimensions: {d}\n'
        assert d[0] == 1 or d[1] == 1, f'\nOne Dimension must be 1: {d}\n'
            
        layerOutputs = networkInputs.copy() if Dimensions( networkInputs )[1] == 1 else transpose(networkInputs)
        
        layerForwardOutputs = [layerOutputs]
        layerForwardDerivatives = []
                    
        for i in range( 0, self.layersCount ) :
            
            layerOutputs, layerDerivatives = self.ForwardPropSingleLayer( layerOutputs, i )
            if self.verbose : print(f'\nlayerOutputs: {layerOutputs}\n')
            
            if self.verbose == 2 : print(f'LayerInformation: {list( zip( layerOutputs, layerDerivatives ) )}\n')
            
            layerForwardOutputs.append( layerOutputs )
            layerForwardDerivatives.append( layerDerivatives )
            
        if train :
            self.history.append([])
            self.history[ self.iterationCount ].append( {'layerOutputs':layerOutputs, 'layerForwardOutputs':layerForwardOutputs, 'layerForwardDerivatives':layerForwardDerivatives} )
        
            self.iterationCount+=1
        else :
            return layerOutputs

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
                    
                    for inNodeW in range( 0, inputDims[1] ) :
                            if self.verbose == 2 : print(f'\t\tinNodeW={inNodeW}', end='')
                    
                            stmt1 = inputsToLayer[inNode][inNodeW]
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
        
        # print(f'trues:{trues}')
        # self.verbose = 2
                    
        outputs = list( zip( *[ history[0]["layerForwardOutputs"].copy() for history in self.history[-len(trues):][:] ] ) )
        
        derivatives = [ history[0]["layerForwardDerivatives"].copy() for history in self.history[-len(trues):][:] ][:]

        weights = self.weights.copy()        
    
        outputs.reverse()
        [ derv.reverse() for derv in derivatives ]    
        weights.reverse()
                                            
        predictions = outputs[0]
                
        if self.verbose == 2 : print( f'Predictions: {predictions}\nOutputs: {outputs}\n\nDerivatives: {derivatives}\nCurrentWeights: {self.weights}\n')
        
        losses_der = list( zip( self.calculateLossOfBatch( trues, outputs ), derivatives[0] ) )
                
        if self.verbose == 2 : print(f'losses_der:{losses_der}')
        
        dLoss = [ 0 for _ in range( len( losses_der ) ) ]
        
        def getFirstLayerLoss() :
        
            # print(f'\t{losses_der[0]}')
            # print(f'\tdLoss:{dLoss}')
            
            loss = losses_der[0][0]
            # print(f'\tLoss:{loss}')
            
            for j, derv in enumerate( losses_der[0][1] ):
                # print(f'\t\t{loss}\t{derv}')
                dLoss[j] += ( loss * derv[0] )
        
        getFirstLayerLoss()
        
        dL_da = [dLoss]
        
        # print(f'\ndl_da {dL_da}\n')
        # print(f'{len(self.weights)}:Weights:{self.weights}')
        # print(f'{len(weights)}:Weights:{weights}')
        # print(f'Derivatives:{derivatives}')
        
                
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
                    if self.verbose == 2 : print(f'\t\tCrossProd : {cp}')
                    
                    batchSumDervs = [ der[ layer+1 ][ connectingWeight ][0] for der in derivatives ]
                    if self.verbose == 2 : print(f'\t\tBatchSummedDervs : {batchSumDervs}')
                                        
                    dL_da[ layer+1 ][ connectingWeight ] = cp * sum( batchSumDervs )
                    # dL_da[ layer+1 ][ connectingWeight ] = cp * derivatives[ layer+1 ][ connectingWeight ][0]
                    
        
        if self.verbose : print(f'\ndl_da {dL_da}\n')
        self.dL_da = dL_da
        
        # self.verbose = 0
    
    def updateWeights( self ) :
        
        outputs = [ history[0]["layerForwardOutputs"].copy() for history in self.history[-self.batchSize:][:] ]
        [ output.reverse() for output in outputs]
        
        
        weights = self.weights.copy()
        weights.reverse()
    
        if self.verbose == 2 : print(f'\toutputs: {outputs}\n\tPartials: {self.dL_da}\n\tLearning Rate: {self.lr}')
        
        gradients = []

        for i, output in enumerate( outputs ) :
            if self.verbose == 2 : print(f'\nBatch Item #{i+1}')
            
            gradients.append([])
            
            for layer in range( 0, len( weights ) ) :
                
                if self.verbose == 2 : print(f'\n\t\tPartial @ layer {layer} : {self.dL_da[layer]}')
                if self.verbose == 2 : print(f'\t\tOutput @ layer {layer+1} : {output[layer+1]}')
                
                def allCombinations( l1, l2 ) :
                    
                    # print(f'l1:{l1}')
                    # print(f'l2:{l2}')
                    
                    ret = []
                                        
                    for j in l1 :
                        # print(f'j:{j}')
                        
                        for i in l2 :
                            # print(f'i:{i}')
                            
                            ret.append( i[0] * j )
                    
                    # print(ret)
                    
                    return ret
                
                gradients[i].append( allCombinations( self.dL_da[layer], output[layer+1] ) )

        if self.verbose :
            print(f'\nGradients:')
            [ print( f'\t{grad}' ) for grad in gradients ]
            print()
            
        for i, gradient in enumerate( gradients ) :  
            if self.verbose == 2 : print(f'\nBatch Item #{i+1}')      
        
            for layer in range( 0, len( weights ) ) :
                
                if self.verbose == 2 : print(f'\n\tweights[{layer}] = {weights[layer]}\n\tWeight Gradients: {gradient[layer]}\n')
                
                idx = 0
                
                for node in range( 0, len( weights[layer] ) ) :
                    if self.verbose == 2 : print(f'\t\tnode[{node}] = {weights[layer][node]}\n')
                    
                    for connectionWeight in range( 0, len( weights[layer][node] ) ) :
                        # if self.verbose == 2 : print(f'\t\tconnection[{connectionWeight}] = {weights[layer][node][connectionWeight]}')
                        
                        # if self.verbose == 2 : print(f'\t\t\tGradient[{layer}][{idx}] = {gradients[layer][idx]}\n')
                        
                        if self.verbose == 2 : print(f'\t\t\tidx:{idx}')
                        if self.verbose == 2 : print(f'\t\t\t\tSingle Weight: {weights[layer][node][connectionWeight]}')
                        if self.verbose == 2 : print(f'\t\t\t\tSingle Gradient: {gradient[layer][idx]}')
                        if self.verbose == 2 : print(f'\t\t\t\tNew Weight = {weights[layer][node][connectionWeight] - self.lr*gradient[layer][idx]}')
                        
                        weights[layer][node][connectionWeight] = weights[layer][node][connectionWeight] - self.lr*gradient[layer][idx]
                        idx+=1

        if self.verbose : print(f"\nUpdated Weights: {weights}")
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
    
    def calculateLossOfBatch( self, trues, outputs ) :
        
        # print(f'trues:{trues}')
        # print(f'outputs:{outputs[0]}')
                
        batchHistory = self.history[-len(trues):]
        
        losses = []
        iterationLoss = 0
        
        for i in range( len( batchHistory ) ) :

            if self.verbose : print( f'Iteration {i+1} output : {outputs[0][i]}' )

            losses.append( self.lossFunc[1]( trues[i], outputs[0][i], verbose=self.verbose ) )
            iterationLoss += sum( self.lossFunc[0]( trues[i], outputs[0][i], verbose=0 ) )
        
        self.iterationLoss = iterationLoss / len( batchHistory )
                
        losses = list( zip( *losses ) )
        if self.verbose == 2 : print( f'\nLosses Zipped : {losses}' )
                
        batchLoss = []
        
        for i in range( len( losses ) ) :
            batchLoss.append(0)
            for j in range( len( losses[0] ) ) :
                batchLoss[ i ] += losses[i][j]
        
        batchLoss = [ loss / len( trues ) for loss in batchLoss ]

        if self.verbose : print(f'\tBatch dLoss : {batchLoss}')
        
        return batchLoss
        # self.batchL/oss = batchLoss
    
    def goDownALayer( self ) : self.currentLayer+=1
    
    def goUpALayer( self ) : self.currentLayer-=1
    
    def setInputDim( self, numberOfInputs:int ) :
        self.inputDim = [ numberOfInputs, 1 ]
        
    def setLossFunc( self, lossFunc:str ) :
        self.lossFunc = self.lossFuncDict[ lossFunc ]
        
    def setLearningRate( self, lr:float ) :
        self.lr = lr
    

