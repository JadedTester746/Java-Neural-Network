import java.util.*;
import java.io.*;
public class DigitRecognizer
{
    public static void main(String[] args){
        double[][] TrainingData = new double[60000][784];
        double[][] ExpectedOutputs = new double[60000][10];
        double LearningRate = 0.07;
        int Epoches = 10000;
        int BatchSize = 32;
        float Cost = 0;
        float LowestCost = 100000000;
        try { 
            Scanner scan = new Scanner(new File("C:\\Users\\danie\\Desktop\\NueralNetwork\\mnist_train.csv"));
            scan.next();
            for(int i = 0; scan.hasNext(); i++){
                String[] Image = scan.next().split(",");
                
                if(Image[0].equals("0")){
                    ExpectedOutputs[i] = new double[] { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
                }
                else if(Image[0].equals("1")){
                    ExpectedOutputs[i] = new double[] {0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
                }
                else if(Image[0].equals("2")){
                    ExpectedOutputs[i] = new double[] {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
                }
                else if(Image[0].equals("3")){
                    ExpectedOutputs[i] = new double[] {0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
                }
                else if(Image[0].equals("4")){
                    ExpectedOutputs[i] = new double[] {0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
                }
                else if(Image[0].equals("5")){
                    ExpectedOutputs[i] = new double[] {0, 0, 0, 0, 0, 1, 0, 0, 0, 0};
                }
                else if(Image[0].equals("6")){
                    ExpectedOutputs[i] = new double[] {0, 0, 0, 0, 0, 0, 1, 0, 0, 0};
                }
                else if(Image[0].equals("7")){
                    ExpectedOutputs[i] = new double[] {0, 0, 0, 0, 0, 0, 0, 1, 0, 0};
                }
                else if(Image[0].equals("8")){
                    ExpectedOutputs[i] = new double[] {0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
                }
                else{
                    ExpectedOutputs[i] = new double[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
                }
                for(int j = 0; j < 784; j++){
                    
                    TrainingData[i][j] = Double.parseDouble(Image[j+1])/255.0;
                    
                }
                
            }
            scan.close();
        }
        catch(Exception e){System.out.println("Can't find File");}
        int[][] size = {{5, 5, 5, 5, 5, 5, 5, 5}, {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}};
        CNN cnn = new CNN(size, 28, new CustomNeuralNetwork(800, 400, 2, 10));    
        cnn.initilize(-1, 1);
        double[] lrs = {0.001, 0.01, 0.02, 0.03, 0.04};
        int[] batchs = {2, 128, 512, 16};
        for(int i = 0; i < lrs.length; i++){
            for(int j = 0; j < batchs.length; j++){
                System.out.println("Final output: " +  TrainNetwork(cnn, Epoches, batchs[j], TrainingData, lrs[i], ExpectedOutputs) + "LR: " + lrs[i] + " Batches:" + batchs[j]);
            }
        }
        
    }
    
    public static String TrainNetwork(CNN network, int Epoches, int BatchSize, double[][] TrainingData, double LearningRate, double[][] ExpectedOutputs){
        network.initilize(-1, 1);
        Random rand = new Random();
        ConvolutionDelta delta = new ConvolutionDelta();
        for(int i = 0; i < Epoches; i++){
            for(int j = 0; j < 60000; j+= BatchSize){
                for(int k = j; k < j + BatchSize; k++){
                    ConvolutionalPrediction guess = network.forward(CNN.makeImage(TrainingData[k], 28, 0));
                    guess.neuralPrediction.Expected = ExpectedOutputs[k];
                    if(k==j){delta = network.backPropagate(guess);}
                    else{delta.add(network.backPropagate(guess));}
                }
                
                delta.average(BatchSize);
                network.applyGradients(delta, BatchSize, LearningRate);
            }
            if(i % 10 == 0){
                System.out.println(Test(network, TrainingData, ExpectedOutputs));
            }
        }
        
        
        
        
        return Test(network, TrainingData, ExpectedOutputs);
    }
    
    public static int Max(double[] x){
        int Max = 0;
        for(int i = 0; i < x.length; i++){
            if(x[Max] < x[i]){
                Max = i;
            } 
        }
        return Max;
    }
    
    public static String Test(CNN network, double[][] TrainingData, double[][] ExpectedOutputs){
        int Correct = 0;
        try{
            
            Scanner scan = new Scanner(new File("C:\\Users\\danie\\Desktop\\NueralNetwork\\mnist_test.csv"));
            scan.next();
            for(int i = 0; i < 10000; i++){
                String[] Image = scan.next().split(",");
                double[] Outputs = new double[10];
                double[] TestingData = new double[784];
                if(Image[0].equals("0")){
                    Outputs = new double[] { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
                }
                else if(Image[0].equals("1")){
                    Outputs = new double[] {0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
                }
                else if(Image[0].equals("2")){
                    Outputs = new double[] {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
                }
                else if(Image[0].equals("3")){
                    Outputs = new double[] {0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
                }
                else if(Image[0].equals("4")){
                    Outputs = new double[] {0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
                }
                else if(Image[0].equals("5")){
                    Outputs = new double[] {0, 0, 0, 0, 0, 1, 0, 0, 0, 0};
                }
                else if(Image[0].equals("6")){
                    Outputs = new double[] {0, 0, 0, 0, 0, 0, 1, 0, 0, 0};
                }
                else if(Image[0].equals("7")){
                    Outputs = new double[] {0, 0, 0, 0, 0, 0, 0, 1, 0, 0};
                }
                else if(Image[0].equals("8")){
                    Outputs = new double[] {0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
                }
                else{
                    Outputs = new double[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
                }
                for(int j = 0; j < 784; j++){
                    
                    TestingData[j] = Double.parseDouble(Image[j+1])/255.0 ;
                    
                }
                double[][] image = CNN.makeImage(TestingData, 28, 0);
                ConvolutionalPrediction out = network.forward(image);
                if(Max(out.neuralPrediction.Outputs) == Max(Outputs)){
                    Correct++;
                }
                //System.out.println(Max(Outputs));
            }
        }
        catch(Exception e){}
        String testing = "" + Correct/10000.0;
        Correct = 0;
        for(int i = 0; i < TrainingData.length; i++){
            double[][] image = CNN.makeImage(TrainingData[i], 28, 0);
            ConvolutionalPrediction out = network.forward(image);
            if(Max(out.neuralPrediction.Outputs) == Max(ExpectedOutputs[i])){
                Correct++;
            }
        }
        return testing + (" Training:" + (double)(Correct)/TrainingData.length);
    }
    
    public static double Test2(CNN[] network){
        int Correct = 0;
        try{
            
            Scanner scan = new Scanner(new File("C:\\Users\\danie\\Desktop\\NueralNetwork\\mnist_test.csv"));
            scan.next();
            for(int i = 0; i < 10000; i++){
                String[] Image = scan.next().split(",");
                double[] Outputs = new double[10];
                double[] TestingData = new double[784];
                if(Image[0].equals("0")){
                    Outputs = new double[] { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
                }
                else if(Image[0].equals("1")){
                    Outputs = new double[] {0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
                }
                else if(Image[0].equals("2")){
                    Outputs = new double[] {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
                }
                else if(Image[0].equals("3")){
                    Outputs = new double[] {0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
                }
                else if(Image[0].equals("4")){
                    Outputs = new double[] {0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
                }
                else if(Image[0].equals("5")){
                    Outputs = new double[] {0, 0, 0, 0, 0, 1, 0, 0, 0, 0};
                }
                else if(Image[0].equals("6")){
                    Outputs = new double[] {0, 0, 0, 0, 0, 0, 1, 0, 0, 0};
                }
                else if(Image[0].equals("7")){
                    Outputs = new double[] {0, 0, 0, 0, 0, 0, 0, 1, 0, 0};
                }
                else if(Image[0].equals("8")){
                    Outputs = new double[] {0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
                }
                else{
                    Outputs = new double[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
                }
                for(int j = 0; j < 784; j++){
                    
                    TestingData[j] = Double.parseDouble(Image[j+1])/255.0 ;
                    
                }
                double[][] image = CNN.makeImage(TestingData, 28, 0);
                // ConvolutionalPrediction out = network.forward(image);
                double[] guess = new double[10];
                for(int j = 0; i < network.length; i++){
                    double[] temp = network[j].forward(image).neuralPrediction.Outputs;
                    for(int k = 0; k < 10; k++){
                        guess[k] += temp[k];
                    }
                }
                if(Max(guess) == Max(Outputs)){
                    Correct++;
                }
                //System.out.println(Max(Outputs));
            }
        }
        catch(Exception e){}
        return Correct/10000.0;
    }
}
