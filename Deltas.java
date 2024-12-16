

public class Deltas
{
    double[][] OutputWeightDeltas;
    double[] OutputBiasDeltas;
    double[][][] HiddenWeightDeltas;
    double[][] HiddenBiasDeltas;
    double[][] InputWeightsDeltas;
    float Cost;
    public Deltas Add(Deltas x, Deltas y){
        for(int k = 0; k < x.HiddenBiasDeltas.length; k++){
            for(int l = 0; l < x.HiddenBiasDeltas[k].length; l++){
                x.HiddenBiasDeltas[k][l] += y.HiddenBiasDeltas[k][l];
                //x.HiddenBiasDeltas[k][l] /= 2.0;
            }
        }
        for(int k = 0; k < x.HiddenWeightDeltas.length; k++){
            for(int l = 0; l < x.HiddenWeightDeltas[k].length; l++){
                for(int m = 0; m < x.HiddenWeightDeltas[k][l].length; m++){
                     x.HiddenWeightDeltas[k][l][m] += y.HiddenWeightDeltas[k][l][m];
                     //x.HiddenWeightDeltas[k][l][m] /= 2.0;
                }
            }
        }
        for(int k = 0; k < x.InputWeightsDeltas.length; k++){
            for(int l = 0; l < x.InputWeightsDeltas[k].length; l++){
                x.InputWeightsDeltas[k][l] += y.InputWeightsDeltas[k][l];
                //x.InputWeightsDeltas[k][l]/=2.0;
                
            }
        }
        for(int k = 0; k < x.OutputBiasDeltas.length; k++){
            x.OutputBiasDeltas[k] += y.OutputBiasDeltas[k];
            //x.OutputBiasDeltas[k]/=2.0;
        }
        for(int k = 0; k < x.OutputWeightDeltas.length; k++){
            for(int l = 0; l < x.OutputWeightDeltas[k].length; l++){
                x.OutputWeightDeltas[k][l] += y.OutputWeightDeltas[k][l];
                //x.OutputWeightDeltas[k][l] /= 2.0;
            }
        }
        return x;
    }
    
    public Deltas Average(Deltas x, double batches){
        for(int k = 0; k < x.HiddenBiasDeltas.length; k++){
            for(int l = 0; l < x.HiddenBiasDeltas[k].length; l++){
                
                x.HiddenBiasDeltas[k][l] /= batches;
            }
        }
        for(int k = 0; k < x.HiddenWeightDeltas.length; k++){
            for(int l = 0; l < x.HiddenWeightDeltas[k].length; l++){
                for(int m = 0; m < x.HiddenWeightDeltas[k][l].length; m++){
                     
                     x.HiddenWeightDeltas[k][l][m] /= batches;
                }
            }
        }
        for(int k = 0; k < x.InputWeightsDeltas.length; k++){
            for(int l = 0; l < x.InputWeightsDeltas[k].length; l++){
                
                x.InputWeightsDeltas[k][l]/=batches;
                
            }
        }
        for(int k = 0; k < x.OutputBiasDeltas.length; k++){
            
            x.OutputBiasDeltas[k]/=batches;
        }
        for(int k = 0; k < x.OutputWeightDeltas.length; k++){
            for(int l = 0; l < x.OutputWeightDeltas[k].length; l++){
                
                x.OutputWeightDeltas[k][l] /= batches;
            }
        }
        return x;
    }
}
