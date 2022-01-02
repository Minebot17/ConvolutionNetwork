package ru.minebot.Convolution;

import ru.minebot.Utils;

import java.util.ArrayList;
import java.util.List;

public class Filter {
    private List<FloatMatrix> matrixes = new ArrayList<>();
    public List<FloatMatrix> lastChange = new ArrayList<>();
    private int width;
    private int height;

    public Filter(int width, int height, int depth){
        this.width = width;
        this.height = height;
        for (int i = 0; i < depth; i++)
            matrixes.add(Utils.randomMatrix(width, height));
        for (int i = 0; i < depth; i++)
            lastChange.add(new FloatMatrix(width, height));
    }

    public FloatMatrix applyFilter(List<FloatMatrix> input, int stride){
        FloatMatrix result = new FloatMatrix((input.get(0).getWidth() - width + 1)/stride, (input.get(0).getHeight() - height + 1)/stride);
        for(int x = 0; x < result.getWidth(); x+=stride)
            for(int y = 0; y < result.getHeight(); y+=stride){
                float value = 0;
                for (int d = 0; d < matrixes.size(); d++)
                    for(int xF = 0; xF < width; xF++)
                        for(int yF = 0; yF < height; yF++)
                            value += input.get(d).getElement(x + xF, y + yF) * matrixes.get(d).getElement(xF, yF);
                result.setElement(x/stride, y/stride, value);
            }
        return result;
    }

    public int getWidth(){
        return width;
    }

    public int getHeight(){
        return height;
    }

    public int getDepth(){
        return matrixes.size();
    }

    public FloatMatrix getMatrix(int index){
        return matrixes.get(index);
    }

    public List<String> serialize(){
        List<String> result = new ArrayList<>();
        result.add(width+"");
        result.add(height+"");
        float[] toAdd = Utils.convertToArray(matrixes);
        for (int i = 0; i < toAdd.length; i++)
            result.add(toAdd[i]+"");
        return result;
    }

    public void deserialize(List<String> data){
        width = Integer.parseInt(data.get(0));
        height = Integer.parseInt(data.get(1));
        float[] array = new float[data.size() - 2];
        for (int i = 0; i < array.length; i++)
            array[i] = Float.parseFloat(data.get(i + 2));
        matrixes = Utils.convertToMatrixes(array, width, height);
    }
}
