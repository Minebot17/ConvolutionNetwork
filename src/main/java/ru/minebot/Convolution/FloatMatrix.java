package ru.minebot.Convolution;

import ru.minebot.Utils;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class FloatMatrix {
    private float[][] matrix;

    public FloatMatrix(float[][] matrix){
        this.matrix = matrix;
    }

    public FloatMatrix(int width, int height){
        this.matrix = new float[width][];
        for (int i = 0; i < width; i++){
            float[] buffer = new float[height];
            for (int j = 0; j < height; j++)
                buffer[j] = Utils.rnd.nextFloat();
            matrix[i] = buffer;
        }
    }

    public void addPadding(int padding){
        float[][] newMatrix = new float[getWidth() + padding*2][];
        newMatrix[0] = new float[getHeight() + padding*2];
        newMatrix[getWidth() + padding] = new float[getHeight() + padding*2];
        for (int i = padding; i < getWidth() + padding; i++){
            float[] newRow = new float[getWidth() + padding*2];
            for (int j = padding; j < getWidth() + padding; j++)
                newRow[j] = getElement(i - padding, j - padding);
            newMatrix[i] = newRow;
        }
        matrix = newMatrix;
    }

    public float getElement(int x, int y){
        return matrix[x][y];
    }

    public void setElement(int x, int y, float value){
        matrix[x][y] = value;
    }

    public void add(int x, int y, float value){ matrix[x][y] += value; }

    public int getWidth(){
        return matrix.length;
    }

    public int getHeight(){
        return matrix[0].length;
    }

    public List<Float> asList(){
        List<Float> result = new ArrayList<>();
        for(int x = 0; x < getWidth(); x++)
            for(int y = 0; y < getHeight(); y++)
                result.add(getElement(x, y));
        return result;
    }

    public FloatMatrix getSubMatrix(int xOffset, int yOffset, int width, int height){
        float[][] result = new float[width][];
        for (int i = 0; i < width; i++){
            float[] toAdd = new float[height];
            for (int j = 0; j < height; j++)
                toAdd[j] = matrix[i + xOffset][j + yOffset];
            result[i] = toAdd;
        }
        return new FloatMatrix(result);
    }

    public FloatMatrix multiply(FloatMatrix matrix){
        FloatMatrix result = new FloatMatrix(getWidth(), getHeight());
        for(int x = 0; x < getWidth(); x++)
            for(int y = 0; y < getHeight(); y++)
                result.setElement(x, y, matrix.getElement(x, y) * getElement(x, y));
        return result;
    }

    public float sum(){
        float result = 0;
        for(int x = 0; x < getWidth(); x++)
            for(int y = 0; y < getHeight(); y++)
                result += getElement(x, y);
        return result;
    }

    public FloatMatrix flip(){
        FloatMatrix result = new FloatMatrix(getWidth(), getHeight());
        for(int x = 0; x < getWidth(); x++)
            for(int y = 0; y < getHeight(); y++)
                result.setElement(getWidth() - x - 1, getHeight() - y - 1, getElement(x, y));
        return result;
    }

    public FloatMatrix copy(){
        return new FloatMatrix(matrix);
    }
}
