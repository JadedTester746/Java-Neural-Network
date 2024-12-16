
public class ConvolutionalLayerDeltas
{
    double[][][] BiasDelta;
    double[][][][] KernelDelta;
    double[][][] imageGradients;
    
    public void add(ConvolutionalLayerDeltas a){
        for(int i = 0; i < BiasDelta.length; i++){
            for(int j = 0; j < BiasDelta[i].length; j++){
                for(int k = 0; k < BiasDelta[i][j].length; k++){
                    BiasDelta[i][j][k] += a.BiasDelta[i][j][k];
                }
            }
        }
        
        for(int i = 0; i < KernelDelta.length; i++){
            for(int j = 0; j < KernelDelta[i].length; j++){
                for(int k = 0; k < KernelDelta[i][j].length; k++){
                    for(int l = 0; l < KernelDelta[i][j][k].length; l++){
                        KernelDelta[i][j][k][l] += a.KernelDelta[i][j][k][l];
                    }
                }
            }
        }

        // Add imageGradients
        for(int i = 0; i < imageGradients.length; i++){
            for(int j = 0; j < imageGradients[i].length; j++){
                for(int k = 0; k < imageGradients[i][j].length; k++){
                    imageGradients[i][j][k] += a.imageGradients[i][j][k];
                }
            }
        }
    }
    
    public void Average(int batchSize){
         for(int i = 0; i < BiasDelta.length; i++){
            for(int j = 0; j < BiasDelta[i].length; j++){
                for(int k = 0; k < BiasDelta[i][j].length; k++){
                    BiasDelta[i][j][k] /= batchSize;
                }
            }
        }
        
        for(int i = 0; i < KernelDelta.length; i++){
            for(int j = 0; j < KernelDelta[i].length; j++){
                for(int k = 0; k < KernelDelta[i][j].length; k++){
                    for(int l = 0; l < KernelDelta[i][j][k].length; l++){
                        KernelDelta[i][j][k][l] /= batchSize;
                    }
                }
            }
        }

        // Add imageGradients
        for(int i = 0; i < imageGradients.length; i++){
            for(int j = 0; j < imageGradients[i].length; j++){
                for(int k = 0; k < imageGradients[i][j].length; k++){
                    imageGradients[i][j][k] /= batchSize;
                }
            }
        }
    }
}
