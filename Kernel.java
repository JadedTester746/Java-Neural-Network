
public class Kernel
{
    public double[][] weights;
    
    public double[][] convolution(double[][] data, int stride){
        double[][] out = new double[1 + ((data.length - weights.length)/stride)][1 + ((data[0].length - weights[0].length)/stride)];
        for(int r = 0, index1 = 0; r < data.length - weights.length + 1; r+= stride){
            for(int c = 0, index2 = 0; c < data[0].length - weights[0].length + 1; c+= stride){
                double sum = 0;
                for(int i = r; i < r + weights.length; i++){
                    
                    for(int j = c; j < c + weights[0].length; j++){
                        sum += data[i][j] * weights[i - r][j - c];
                    }
                }
                out[index1][index2] = sum;
                index2++;
            }
            index1++;
        }
        return out;
    }
    
    public Kernel(double[][] in){
        weights = in;
    }
    
    
    public double[][] fullConvolution(double[][] image) {
        int imageHeight = image.length;
        int imageWidth = image[0].length;
        int weightsHeight = weights.length;
        int weightsWidth = weights[0].length;
        
        int resultHeight = imageHeight + weightsHeight - 1;
        int resultWidth = imageWidth + weightsWidth - 1;
        
        double[][] result = new double[resultHeight][resultWidth];
        
        // Flip the weights
  
        
        // Perform convolution
        for (int i = 0; i < resultHeight; i++) {
            for (int j = 0; j < resultWidth; j++) {
                double sum = 0;
                for (int ki = 0; ki < weightsHeight; ki++) {
                    for (int kj = 0; kj < weightsWidth; kj++) {
                        int imageRowIndex = i - ki;
                        int imageColIndex = j - kj;
                        if (imageRowIndex >= 0 && imageRowIndex < imageHeight &&
                            imageColIndex >= 0 && imageColIndex < imageWidth) {
                            sum += image[imageRowIndex][imageColIndex] * weights[ki][kj];
                        }
                    }
                }
                result[i][j] = sum;
            }
        }
        
        return result;
    }
}
