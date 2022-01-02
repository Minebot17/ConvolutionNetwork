package ru.minebot;

import ru.minebot.Convolution.FloatMatrix;
import ru.minebot.Forward.ForwardNetwork;
import ru.minebot.Forward.Neuron;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Utils {
    public static final Random rnd = new Random();
    private static final double EPSILON = 1e-10;

    // Gaussian elimination with partial pivoting
    public static float[] gaussSolve(float[][] A, float[] b) {
        int n = b.length;

        for (int p = 0; p < n; p++) {

            // find pivot row and swap
            int max = p;
            for (int i = p + 1; i < n; i++) {
                if (Math.abs(A[i][p]) > Math.abs(A[max][p])) {
                    max = i;
                }
            }
            float[] temp = A[p];
            A[p] = A[max];
            A[max] = temp;
            float t = b[p];
            b[p] = b[max];
            b[max] = t;

            // singular or nearly singular
            if (Math.abs(A[p][p]) <= EPSILON) {
                throw new ArithmeticException("Matrix is singular or nearly singular");
            }

            // pivot within A and b
            for (int i = p + 1; i < n; i++) {
                double alpha = A[i][p] / A[p][p];
                b[i] -= alpha * b[p];
                for (int j = p; j < n; j++) {
                    A[i][j] -= alpha * A[p][j];
                }
            }
        }

        // back substitution
        float[] x = new float[n];
        for (int i = n - 1; i >= 0; i--) {
            float sum = 0.0f;
            for (int j = i + 1; j < n; j++) {
                sum += A[i][j] * x[j];
            }
            x[i] = (b[i] - sum) / A[i][i];
        }
        return x;
    }

    public static float[] pickInputs(float[] cofs, float[][] weightMatrix, int iters){
        Random rnd = new Random();
        float[] result = null;
        float minError = 999999999;
        for (int i = 0; i < iters; i++) {
            float[] gen = new float[weightMatrix[0].length];
            for (int j = 0; j < gen.length; j++)
                gen[j] = rnd.nextFloat();

            float error = 0;
            for (int j = 0; j < weightMatrix.length; j++){
                float sum = 0;
                for (int m = 0; m < weightMatrix[j].length; m++)
                    sum += weightMatrix[j][m] * gen[m];
                error += Math.pow(sum - cofs[j], 2);
            }
            if (error < minError){
                minError = error;
                result = gen;
            }
        }
        return result;
    }

    public static ForwardNetwork generateSimpleForwardNetwork(float learnSpeed, float momentum, int[] layersNeuronsCount, boolean withBias){
        Neuron[][] result = new Neuron[layersNeuronsCount.length][];
        for (int i = 0; i < layersNeuronsCount.length; i++) {
            result[i] = new Neuron[layersNeuronsCount[i]];
            for (int j = 0; j < layersNeuronsCount[i]; j++)
                result[i][j] = new Neuron(i, j, ActivationFunctions.SIGMOID);
        }
        return new ForwardNetwork(learnSpeed, momentum, withBias, result);
    }

    public static void delete(File f) throws IOException {
        if (f.isDirectory()) {
            for (File c : f.listFiles())
                delete(c);
        }
        if (!f.delete())
            throw new FileNotFoundException("Failed to delete file: " + f);
    }

    public static List<FloatMatrix> prepareImage(BufferedImage image){
        List<FloatMatrix> result = new ArrayList<>();
        int i = 0;
        FloatMatrix m = new FloatMatrix(image.getWidth(), image.getHeight());
        for (int x = 0; x < image.getWidth(); x++)
            for (int y = 0; y < image.getHeight(); y++)
                m.setElement(x, y, 255 - (image.getRGB(x, y) >> (i == 0 ? 16 : i == 1 ? 8 : 0)) & 255);
        result.add(m);
        return result;
    }

    public static FloatMatrix randomMatrix(int width, int height){
        FloatMatrix result = new FloatMatrix(width, height);
        for(int x = 0; x < width; x++)
            for(int y = 0; y < height; y++)
                result.setElement(x, y, rnd.nextFloat());
        return result;
    }

    public static float[] convertToArray(List<FloatMatrix> matrices){
        List<Float> result = new ArrayList<>();
        for (int i = 0; i < matrices.size(); i++)
            result.addAll(matrices.get(i).asList());
        float[] array = new float[result.size()];
        for (int i = 0; i < array.length; i++)
            array[i] = result.get(i);
        return array;
    }

    public static List<FloatMatrix> convertToMatrixes(float[] array, int width, int height){
        List<FloatMatrix> result = new ArrayList<>();
        for (int i = 0; i < array.length/(width * height); i++) {
            FloatMatrix matrix = new FloatMatrix(width, height);
            for (int x = 0; x < width; x++)
                for (int y = 0; y < height; y++)
                    matrix.setElement(x, y, array[x * width + y]);
            result.add(matrix);
        }
        return result;
    }
}
