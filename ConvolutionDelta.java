
public class ConvolutionDelta
{
    Deltas delta;
    ConvolutionalLayerDeltas[] layerDeltas;
    
    public void add(ConvolutionDelta a){
        delta = delta.Add(delta, a.delta);
        for(int i = 0; i < layerDeltas.length; i++){
            layerDeltas[i].add(a.layerDeltas[i]);
        }
    }
    public void average(int batchSize){
        delta = delta.Average(delta, batchSize);
        for(int i = 0; i < layerDeltas.length; i++){
            layerDeltas[i].Average(batchSize);
        }

    }
}
