import java.util.*;
public class CNN
{
    public CustomNeuralNetwork network;
    public ConvolutionalLayer[] layers;
    
    public ConvolutionalPrediction forward(double[][] image){
        //preproccessing
        double[][][] images = {image};
        ConvolutionalPrediction out = new ConvolutionalPrediction();
        out.layers = new ConvolutionLayerPrediction[layers.length];
        for(int i = 0; i < layers.length; i++){
            ConvolutionLayerPrediction output = layers[i].predict(images);
            images = output.imagesOutput;
            out.layers[i] = output;
        }
        out.neuralPrediction = network.Predict(flatten(images));
        return out;
    }
    
        public static double[] flatten(double[][][] matrix3D) {
        List<Double> flattenedList = new ArrayList<>();

        for (double[][] matrix2D : matrix3D) {
            for (double[] row : matrix2D) {
                for (double num : row) {
                    flattenedList.add(num);
                }
            }
        }

        // Convert List<Double> to double[]
        double[] flattenedArray = new double[flattenedList.size()];
        for (int i = 0; i < flattenedList.size(); i++) {
            flattenedArray[i] = flattenedList.get(i);
        }

        return flattenedArray;
    }
    
    public CNN(int[][] size, int startSize, CustomNeuralNetwork network){ // int[layers][kernels]
        layers = new ConvolutionalLayer[size.length];
        for(int i = 0; i < size.length; i++){
            layers[i] = new ConvolutionalLayer();
            
            if(i == 0){layers[i].Kernels = new Kernel[size[i].length][1];}
            else{layers[i].Kernels = new Kernel[size[i].length][layers[i-1].Kernels.length];}
            
            for(int j = 0; j < layers[i].Kernels.length; j++){
                for(int k = 0; k < layers[i].Kernels[j].length; k++){
                    layers[i].Kernels[j][k] = new Kernel(new double[size[i][0]][size[i][0]]);
                }
            }
            startSize = (1 + (startSize - size[i][0]));
            layers[i].Bias = new double[layers[i].Kernels.length][startSize][startSize];
            startSize /= 2;
            layers[i].cnn = this;
        }
        this.network = network;
    }
    
    public static void main(String[] args){
        int[][] size = {{5, 5}, {3, 3, 3, 3}};
        CNN temp = new CNN(size, 28, new CustomNeuralNetwork(100, 28, 2, 10));
        temp.initilize(-1, 1);
        double[][] image = new double[28][28];
        ConvolutionalPrediction output = temp.forward(image);
        double[] expected = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
        output.neuralPrediction.Expected = expected;
        ConvolutionDelta num = temp.backPropagate(output);
        
        //for(int i = 0; i < output.length; i++){
            //System.out.println(output[i]);
        //}
    }
    
    public void initilize(double min, double max){
        network.InitilizeNetwork(min, max);
        for(ConvolutionalLayer layer : layers){
            layer.initialize(min, max);
        }
    }
    
    public ConvolutionDelta backPropagate(ConvolutionalPrediction output){
        ConvolutionDelta out = new ConvolutionDelta();
        out.delta = network.BackPropagation(output.neuralPrediction);
        
        double[] δ = network.Mult(network.ActivationPrime(network.Add(network.multiply(network.InputWeights, output.neuralPrediction.Inputs), network.HiddenBias[0])), network.Influence(network.InputWeights, out.delta.HiddenBiasDeltas[0]));
        
        double[][][] image = new double[layers[layers.length - 1].Kernels.length][][];
        int start = 0;
        for(int i = 0; i < image.length; i++){
            image[i] = makeImage(δ, output.layers[output.layers.length - 1].imagesOutput[0].length, start);
            start += output.layers[output.layers.length - 1].imagesOutput[0].length * output.layers[output.layers.length - 1].imagesOutput[0].length;
        }
        // it is time
        out.layerDeltas = new ConvolutionalLayerDeltas[layers.length];
        for(int  i = layers.length - 1; i >= 0; i--){
            out.layerDeltas[i] = layers[i].backprop(output.layers[i], image, output);
            image = out.layerDeltas[i].imageGradients;
        }
        // and that, was backpropagation(shockingly simple(possible foreshadowing?))
        return out;
    }
    
    public static double[][] makeImage(double[] data, int length, int start){
        double[][] out = new double[length][length];
        for(int i = start; i < data.length && i - start < length * length; i++){
            out[(i - start) / length][(i - start) % length] = data[i];
        }
        return out;
    }
    
    public void applyGradients(ConvolutionDelta delta, int batchSize, double learningRate){
        network.ApplyGradients(delta.delta, batchSize, 0.07, network);
        
        for(int i = 0; i < layers.length; i++){
            layers[i].applyGradients(delta.layerDeltas[i], batchSize, learningRate);
        }
    }
}




