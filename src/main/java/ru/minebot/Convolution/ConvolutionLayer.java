package ru.minebot.Convolution;

import ru.minebot.ActivationFunctions;

import java.util.ArrayList;
import java.util.List;

public class ConvolutionLayer {

    private List<Filter> filters = new ArrayList<>();
    private List<FloatMatrix> lastInput = null;
    private List<FloatMatrix> lastResult = null;
    private int filtersDepth;

    public ConvolutionLayer(int filtersCount, int filtersSize, int filtersDepth){
        this.filtersDepth = filtersDepth;
        for (int i = 0; i < filtersCount; i++)
            filters.add(new Filter(filtersSize, filtersSize, filtersDepth));
    }

    public List<FloatMatrix> convolve(List<FloatMatrix> input, int stride) throws Exception {
        if (filtersDepth != input.size())
            throw new Exception("Input depth != filters depth");

        lastInput = input;
        List<FloatMatrix> result = new ArrayList<>();
        for (int i = 0; i < filters.size(); i++) {
            FloatMatrix toAdd = filters.get(i).applyFilter(input, stride);
            FloatMatrix postPooling = new FloatMatrix(toAdd.getWidth()/2, toAdd.getHeight()/2);
            for(int x = 0; x < postPooling.getWidth(); x++)
                for(int y = 0; y < postPooling.getHeight(); y++)
                    postPooling.setElement(x, y, ActivationFunctions.RELU.invoke(Math.max(Math.max(toAdd.getElement(x*2, y*2+1), toAdd.getElement(x*2+1, y*2)), Math.max(toAdd.getElement(x*2, y*2), toAdd.getElement(x*2+1, y*2+1)))));
            result.add(postPooling);
        }
        lastResult = result;
        return result;
    }

    public List<FloatMatrix> getLastResult(){ return lastResult; }

    public List<FloatMatrix> getLastInput(){
        return lastInput;
    }

    public Filter getFilter(int index){
        return filters.get(0);
    }

    public int getFilterCount(){
        return filters.size();
    }

    public List<String> serialize(){
        List<String> result = new ArrayList<>();
        result.add(filtersDepth+"");
        for (int i = 0; i < filters.size(); i++){
            result.add("filter start");
            result.addAll(filters.get(i).serialize());
            result.add("filter end");
        }
        return result;
    }

    public void deserialize(List<String> data){
        filtersDepth = Integer.parseInt(data.get(0));
        int start = -1;
        for (int i = 1; i < data.size(); i++){
            if (data.get(i).equals("filter start"))
                start = i;
            else if (data.get(i).equals("filter end")){
                Filter filter = new Filter(1, 1, 1);
                filter.deserialize(data.subList(start + 1, i));
                filters.add(filter);
            }
        }
    }
}
