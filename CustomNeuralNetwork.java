import java.util.*;
import java.util.stream.*;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.OutputStreamWriter;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;

public class CustomNeuralNetwork
{
    double[][] InputWeights;
    double[][][] HiddenWeights;
    double[][] OutputWeights;
    double[][] HiddenBias;
    double[] OutputBias;
    
    public static ThreadPoolExecutor executor;
    static {
    executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(8);
    }

    private static class Job implements Runnable {
        private double[][] matA;
        private double[] matB, mat;
        private int i, j;

        public Job(double[][] matA, double[] matB, double[] mat, int i, int j) {
            this.matA = matA;
            this.matB = matB;
            this.mat = mat;
            this.i = i;
            this.j = j;
        }

        @Override
        public void run() {
            double sum = 0;
            for(int k = 0; k < matB.length; k++) {
                sum += matA[i][k] * matB[k];
            }
            mat[i] = sum;
        }
    }

    // public static double[] multiply(double[][] matA, double[] matB) {
        // double[] mat = new double[matA.length];
        // double sum;
        // for(int i = 0; i < matA.length; i++) {
            // for(int j = 0; j < 1; j++) {
                // executor.submit(new Job(matA, matB, mat, i, j));
            // }
        // }
        
        // while(executor.getQueue().size() > 0);
        
        // return mat;
        
    // }
    
    public CustomNeuralNetwork(int InputSize, int HiddenLayerSize, int HiddenLayers, int OutputSize){
        InputWeights = new double[HiddenLayerSize][InputSize];
        HiddenWeights = new double[HiddenLayers - 1][HiddenLayerSize][HiddenLayerSize];
        OutputWeights = new double[OutputSize][HiddenLayerSize];
        HiddenBias = new double[HiddenLayers][HiddenLayerSize];
        OutputBias = new double[OutputSize];
    }
    public static double[] Activation(double[] x){
        double[] Out = new double[x.length];
        for(int i = 0; i < Out.length; i++){
            Out[i] = 1.0/(1 + Math.exp(-x[i]));
        }
        return Out;
    }
    
    public static double[] ActivationPrime(double[] x){
        double[] Out = new double[x.length];
        for(int i = 0; i < Out.length; i++){
            double activation = Out[i] = 1.0/(1 + Math.exp(-x[i]));
            Out[i] = activation * (1 - activation);
        }
        return Out;
    }



    
    public void InitilizeNetwork(double Min, double Max){
        Random r = new Random();
        for(int i = 0; i < InputWeights.length; i++){
            for(int j = 0; j < InputWeights[i].length; j++){
                InputWeights[i][j] = Min + ((Max - Min) * r.nextDouble());
            }
        }
        for(int i = 0; i < HiddenWeights.length; i++){
            for(int j = 0; j < HiddenWeights[i].length; j++){
                for (int k = 0; k < HiddenWeights[i][j].length; k++){
                    HiddenWeights[i][j][k] = Min + ((Max - Min) * r.nextDouble());
                }
            }
        }
        for(int i = 0; i < OutputWeights.length; i++){
            for(int j = 0; j < OutputWeights[i].length; j++){
                OutputWeights[i][j] = Min + ((Max - Min) * r.nextDouble());
            }
        }
        //Bias's don't need to be initilized since they start at 0
    }
    public static double[] multiply(double[][] matrix, double[] vector) {
        int rows = matrix.length;
        int columns = matrix[0].length;
        double[] result = new double[rows];

        for (int row = 0; row < rows; row++) {
            double sum = 0;
            for (int column = 0; column < columns; column++) {
                sum += matrix[row][column]
                        * vector[column];
            }
            result[row] = sum;
        }
        return result;
        
        
    }
    
    
    
    public static double[] MultConst(double[] a, double x){
        double[] Out = new double[a.length];
        for(int i = 0; i < a.length; i++){
            Out[i] = a[i] * x;
        }
        return Out;
    }
    public static double[] Add(double[] a, double[] b){
        double[] Out = new double[a.length];
        for(int i = 0; i < Out.length; i++){
            Out[i] = a[i] + b[i];
        }
        return Out;
    }
    public static double[] Subtract(double[] a, double[] b){
        double[] Out = new double[a.length];
        for(int i = 0; i < Out.length; i++){
            Out[i] = a[i] - b[i];
        }
        return Out;
    }
    public static double[] Mult(double[] a, double[] b){
        double[] Out = new double[a.length];
        for(int i = 0; i < Out.length; i++){
            Out[i] = a[i] * b[i];
        }
        return Out;
    }
    public Prediction Predict(double[] Inputs){
        Prediction Out = new Prediction();
        
        Out.Inputs = Inputs;
        Out.HiddenLayers = new double[HiddenWeights.length + 1][HiddenBias[0].length];
        double[] CurrentLayer = Activation(Add(multiply(InputWeights, Inputs), HiddenBias[0]));
        Out.HiddenLayers[0] = CurrentLayer;
        for(int i = 0; i < HiddenWeights.length; i++){
            CurrentLayer = Activation(Add(multiply(HiddenWeights[i], CurrentLayer), HiddenBias[i + 1]));
            
            Out.HiddenLayers[i + 1] = CurrentLayer;
        }
        Out.Outputs = Activation(Add(multiply(OutputWeights, CurrentLayer), OutputBias));


        return Out;
        
    }
    public Deltas BackPropagation(Prediction Out){
        //δ Important Greek letter 
        double[] δ = MultConst(Mult((Subtract(Out.Outputs, Out.Expected )), ActivationPrime(Add(multiply(OutputWeights, Out.HiddenLayers[Out.HiddenLayers.length - 1]), OutputBias))), 2);
        double[][] OutputWeightDeltas = new double[OutputWeights.length][OutputWeights[0].length];
        double[] OutputBiasDeltas = new double[OutputBias.length];
        double[][][] HiddenWeightDeltas = new double[HiddenWeights.length][HiddenWeights[0].length][HiddenWeights[0][0].length];
        double[][] HiddenBiasDeltas = new double[HiddenBias.length][HiddenBias[0].length];
        double[][] InputWeightsDeltas = new double[InputWeights.length][Out.HiddenLayers.length];
        for(int i = 0; i < OutputWeightDeltas.length; i++){
            for(int j = 0; j < OutputWeightDeltas[i].length; j++){
                OutputWeightDeltas[i][j] = δ[i] * Out.HiddenLayers[Out.HiddenLayers.length - 1][j];
                
            }
        }
        for(int i = 0; i < OutputBiasDeltas.length; i++){
            OutputBiasDeltas[i] = δ[i];
            
        }
        for(int i = HiddenWeights.length - 1; i >= 0 ; i--){
            if(i == HiddenWeights.length - 1){
                δ = Mult(ActivationPrime(Add(multiply(HiddenWeights[i], Out.HiddenLayers[i]), HiddenBias[i + 1])), Influence(OutputWeights, δ));
            }
            else{
                δ = Mult(ActivationPrime(Add(multiply(HiddenWeights[i], Out.HiddenLayers[i]), HiddenBias[i + 1])), Influence(HiddenWeights[i + 1], δ));
            }

            for(int j = 0; j < HiddenWeightDeltas[i].length; j++){
                for(int k = 0; k < HiddenWeightDeltas[i][j].length; k++){
                    HiddenWeightDeltas[i][j][k] = δ[j] * Out.HiddenLayers[i][k];
                    
                }
            }
            for(int j = 0; j < HiddenBiasDeltas[i + 1].length; j++){
                HiddenBiasDeltas[i + 1][j] = δ[j];
                
            }
        }
        δ = Mult(ActivationPrime(Add(multiply(InputWeights, Out.Inputs), HiddenBias[0])), Influence(HiddenWeights[0], δ));
        for(int i = 0; i < InputWeightsDeltas.length; i++){
            for(int j = 0; j < InputWeightsDeltas[i].length; j++){
                InputWeightsDeltas[i][j] = δ[i] * Out.Inputs[j];
                
            }
        }
        for(int i = 0; i < HiddenBias[0].length; i++){
            HiddenBiasDeltas[0][i] = δ[i];
           
        }
        

        Deltas x = new Deltas();
        x.HiddenBiasDeltas = HiddenBiasDeltas;
        x.HiddenWeightDeltas = HiddenWeightDeltas;
        x.InputWeightsDeltas = InputWeightsDeltas;
        x.OutputBiasDeltas = OutputBiasDeltas;
        x.OutputWeightDeltas = OutputWeightDeltas;
        x.Cost = (float)ComputeCost(Out);
        return x;
        
    }
    
    public static double[] Influence(double[][] Weights, double[] δ){
        double[] Out = new double[Weights[0].length];
        for(int n = 0; n < Out.length; n++){
            for(int m = 0; m < Weights.length; m++){
                Out[n] += δ[m] * Weights[m][n];
            }
        
        }
        return Out;
    }   
    
    public void ApplyGradients(Deltas x, int BatchSize, double LearningRate, CustomNeuralNetwork network){
        Deltas Z = new Deltas();
        
        
        for(int k = 0; k < x.HiddenBiasDeltas.length; k++){
            for(int l = 0; l < x.HiddenBiasDeltas[k].length; l++){
                network.HiddenBias[k][l] -= (LearningRate * x.HiddenBiasDeltas[k][l]);
            }
        }
        for(int k = 0; k < x.HiddenWeightDeltas.length; k++){
            for(int l = 0; l < x.HiddenWeightDeltas[k].length; l++){
                for(int m = 0; m < x.HiddenWeightDeltas[k][l].length; m++){
                    network.HiddenWeights[k][l][m] -= (LearningRate*  x.HiddenWeightDeltas[k][l][m]);
                }
            }
        }
        for(int k = 0; k < x.InputWeightsDeltas.length; k++){
            for(int l = 0; l < x.InputWeightsDeltas[k].length; l++){
                network.InputWeights[k][l] -= (LearningRate*  x.InputWeightsDeltas[k][l]);
            }
        }
        for(int k = 0; k < x.OutputBiasDeltas.length; k++){
            network.OutputBias[k] -= (LearningRate * x.OutputBiasDeltas[k]);
        }
        for(int k = 0; k < x.OutputWeightDeltas.length; k++){
            for(int l = 0; l < x.OutputWeightDeltas[k].length; l++){
                network.OutputWeights[k][l] -= (LearningRate * x.OutputWeightDeltas[k][l]);
            }
        }
                
    }
    
    
    public static double ComputeCost(Prediction Out){
        
        double y = 0;
        for(int i = 0; i < Out.Expected.length; i++){
            y+=(Out.Expected[i] - Out.Outputs[i]) * (Out.Expected[i] - Out.Outputs[i]); 
            
        }
        return y;
    }
    
    
}
