package ru.minebot.Convolution;

import ru.minebot.ActivationFunctions;
import ru.minebot.Forward.Weight;
import ru.minebot.Utils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

public class ConvolutionNetwork {

    private int padding;
    private int stride;
    private float learnSpeed;
    private float momentum;
    private List<ConvolutionLayer> layers = new ArrayList<>();

    public ConvolutionNetwork(int[] firstSize, int[] filtersCount, int filtersSize, int padding, int stride, float learnSpeed, float momentum){
        this.padding = padding;
        this.stride = stride;
        this.learnSpeed = learnSpeed;
        this.momentum = momentum;
        layers.add(new ConvolutionLayer(filtersCount[0], filtersSize, firstSize[2]));
        for (int i = 1; i < filtersCount.length; i++)
            layers.add(new ConvolutionLayer(filtersCount[i], filtersSize, filtersCount[i - 1]));
    }

    public ConvolutionNetwork(String name) throws IOException {
        if (!Files.exists(Paths.get("./saves")))
            Files.createDirectory(Paths.get("./saves"));

        String directory = "./saves/" + name;
        File saveFile = new File( directory + ".cn");
        if (!saveFile.exists())
            throw new IOException("Save don't exists");

        try(ZipInputStream zin = new ZipInputStream(new FileInputStream(saveFile))) {
            ZipEntry entry;
            String fileName;
            Files.createDirectory(Paths.get(directory));
            while ((entry = zin.getNextEntry()) != null) {
                fileName = entry.getName();
                FileOutputStream fout = new FileOutputStream(directory + "/" + fileName);
                for (int c = zin.read(); c != -1; c = zin.read()) {
                    fout.write(c);
                }
                fout.flush();
                zin.closeEntry();
                fout.close();
            }
        }
        catch(Exception ex){
            System.out.println(ex.getMessage());
        }

        List<String> infoList = Files.readAllLines(Paths.get(directory + "/info"));
        List<String> weightsList = Files.readAllLines(Paths.get(directory + "/weights"));
        padding = Integer.parseInt(infoList.get(0));
        stride = Integer.parseInt(infoList.get(1));

        int start = -1;
        for (int i = 0; i < weightsList.size(); i++){
            if (weightsList.get(i).equals("layer start"))
                start = i;
            else if (weightsList.get(i).equals("layer end")){
                ConvolutionLayer layer = new ConvolutionLayer(1, 1, 1);
                layer.deserialize(weightsList.subList(start + 1, i));
                layers.add(layer);
            }
        }

        Utils.delete(new File(directory));
    }

    public float[] convolution(List<FloatMatrix> input) throws Exception {
        for (int i = 0; i < input.size(); i++)
            input.get(i).addPadding(padding);
        for (int i = 0; i < layers.size(); i++)
            input = layers.get(i).convolve(input, stride);
        return Utils.convertToArray(input);
    }

    public void correctWeight(float[] deltas){
        List<FloatMatrix> currentDeltas = Utils.convertToMatrixes(deltas, layers.get(layers.size() - 1).getLastResult().get(0).getWidth(), layers.get(layers.size() - 1).getLastResult().get(0).getHeight());
        for (int i = layers.size() - 1; i >= 0; i--){
            List<FloatMatrix> currentInput = layers.get(i).getLastInput();
            List<FloatMatrix> currentOutput = layers.get(i).getLastResult();
            for (int f = 0; f < layers.get(i).getFilterCount(); f++)
                for(int xF = 0; xF < layers.get(i).getFilter(f).getWidth(); xF++)
                    for(int yF = 0; yF < layers.get(i).getFilter(f).getHeight(); yF++)
                        for (int dF = 0; dF < layers.get(i).getFilter(f).getDepth(); dF++){
                            FloatMatrix targetMatrix = currentInput.get(dF).getSubMatrix(xF, yF, currentOutput.get(f).getWidth(), currentOutput.get(f).getHeight());
                            float deltaWeight = currentDeltas.get(f).multiply(targetMatrix).sum() * learnSpeed + momentum * layers.get(i).getFilter(f).lastChange.get(dF).getElement(xF, yF);
                            layers.get(i).getFilter(f).getMatrix(dF).add(xF, yF, deltaWeight);
                            layers.get(i).getFilter(f).lastChange.get(dF).setElement(xF, yF, deltaWeight);
                        }

            for (int f = 0; f < layers.get(i).getFilterCount(); f++)
                for(int xF = 0; xF < layers.get(i).getFilter(f).getWidth(); xF++)
                    for(int yF = 0; yF < layers.get(i).getFilter(f).getHeight(); yF++)
                        for (int dF = 0; dF < layers.get(i).getFilter(f).getDepth(); dF++){
                            layers.get(i).getFilter(f).getMatrix(dF).setElement(xF, yF, ActivationFunctions.SIGMOID.invoke(layers.get(i).getFilter(f).getMatrix(dF).getElement(xF, yF)));
                        }

            if (i != 0){
                List<FloatMatrix> newDeltas = new ArrayList<>();
                for (int j = 0; j < currentInput.size(); j++)
                    newDeltas.add(new FloatMatrix(currentInput.get(j).getWidth(), currentInput.get(j).getHeight()));
                for (int f = 0; f < layers.get(i).getFilterCount(); f++) {
                    int maxX = (currentInput.get(0).getWidth() - layers.get(i).getFilter(f).getWidth() + 1)/2;
                    int maxY = (currentInput.get(0).getHeight() - layers.get(i).getFilter(f).getHeight() + 1)/2;
                    for (int d = 0; d < currentInput.size(); d++)
                        for (int x = 0; x < maxX; x++)
                            for (int y = 0; y < maxY; y++)
                                for(int xF = 0; xF < layers.get(i).getFilter(f).getWidth(); xF++)
                                    for(int yF = 0; yF < layers.get(i).getFilter(f).getHeight(); yF++) {
                                        FloatMatrix flipped = layers.get(i).getFilter(f).getMatrix(d).flip();
                                        newDeltas.get(d).add(x + xF, y + yF, flipped.getElement(xF, yF) * currentDeltas.get(f).getElement(x, y));
                                    }
                }
                for (int j = 0; j < newDeltas.size(); j++)
                    newDeltas.set(j, newDeltas.get(j).multiply(currentInput.get(j)));
                currentDeltas = newDeltas;
            }
        }
    }

    public void save(String name) throws IOException {
        if (!Files.exists(Paths.get("./saves")))
            Files.createDirectory(Paths.get("./saves"));

        String directory = "./saves/" + name;
        File saveFile = new File( directory + ".cn");
        if (saveFile.exists())
            saveFile.delete();

        Files.createDirectory(Paths.get(directory));

        List<String> infoList = new ArrayList<>();
        infoList.add(padding+"");
        infoList.add(stride+"");
        Files.write(Paths.get(directory + "/info"), infoList);

        List<String> weightsList = new ArrayList<>();
        for (int i = 0; i < layers.size(); i++) {
            weightsList.add("layer start");
            weightsList.addAll(layers.get(i).serialize());
            weightsList.add("layer end");
        }
        Files.write(Paths.get(directory+ "/weights"), weightsList);

        try(ZipOutputStream zout = new ZipOutputStream(new FileOutputStream(directory + ".cn"));
            FileInputStream fisInfo = new FileInputStream(directory + "/info");
            FileInputStream fisWeights = new FileInputStream(directory + "/weights");) {

            ZipEntry infoEntry = new ZipEntry("info");
            zout.putNextEntry(infoEntry);
            byte[] buffer0 = new byte[fisInfo.available()];
            fisInfo.read(buffer0);
            zout.write(buffer0);
            zout.closeEntry();

            ZipEntry weightsEntry = new ZipEntry("weights");
            zout.putNextEntry(weightsEntry);
            byte[] buffer1 = new byte[fisWeights.available()];
            fisWeights.read(buffer1);
            zout.write(buffer1);
            zout.closeEntry();
        }
        Utils.delete(new File(directory));
    }
}
