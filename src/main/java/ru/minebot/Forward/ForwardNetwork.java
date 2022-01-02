package ru.minebot.Forward;

import ru.minebot.ActivationFunctions;
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

public class ForwardNetwork {
    private Neuron[][] layers;
    private Neuron[] biases;
    public float learnSpeed;
    public float momentum;
    private float error = 0;
    private int iterations = 0;
    private boolean withBias;

    public ForwardNetwork(float learnSpeed, float momentum, boolean withBias, Neuron[]... layers){
        this.learnSpeed = learnSpeed;
        this.momentum = momentum;
        this.layers = layers;
        this.withBias = withBias;

        for (int i = 0; i < this.layers.length; i++)
            for (int j = 0; j < this.layers[i].length; j++)
            this.layers[i][j].setLayer(i);

        for (int i = 0; i < this.layers.length - 1; i++)
            for (int j = 0; j < this.layers[i].length; j++)
                for (int u = 0; u < this.layers[i + 1].length; u++){
                    Weight weight = new Weight(this.layers[i][j], this.layers[i + 1][u]);
                    this.layers[i][j].connectWithOut(weight);
                    this.layers[i + 1][u].connectWithIn(weight);
                }

        if (withBias){
            biases = new Neuron[layers.length - 1];
            for (int i = 0; i < biases.length; i++) {
                biases[i] = new Neuron(i, -1, ActivationFunctions.LINEAR);
                biases[i].setValueIn(1);
                for (int j = 0; j < layers[i + 1].length; j++){
                    Weight weight = new Weight(this.biases[i], this.layers[i + 1][j]);
                    this.biases[i].connectWithOut(weight);
                    this.layers[i + 1][j].connectWithIn(weight);
                }
            }
        }
    }

    public ForwardNetwork(String name) throws IOException {
        if (!Files.exists(Paths.get("./saves")))
            Files.createDirectory(Paths.get("./saves"));

        String directory = "./saves/" + name;
        File saveFile = new File( directory + ".fn");
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
        this.learnSpeed = Float.parseFloat(infoList.get(0));
        this.momentum = Float.parseFloat(infoList.get(1));
        this.withBias = Boolean.parseBoolean(infoList.get(2));

        layers = new Neuron[infoList.size() - 3][];
        for (int i = 0; i < infoList.size() - 3; i++){
            layers[i] = new Neuron[Integer.parseInt(infoList.get(3 + i))];
            for (int j = 0; j < layers[i].length; j++)
                layers[i][j] = new Neuron(i, j, i == 0 ? ActivationFunctions.LINEAR : ActivationFunctions.SIGMOID);
        }

        if (this.withBias) {
            biases = new Neuron[layers.length - 1];
            for (int i = 0; i < biases.length; i++) {
                biases[i] = new Neuron(i, -1, ActivationFunctions.LINEAR);
                biases[i].setValueIn(1);
            }
        }

        for (int i = 0; i < weightsList.size(); i++){
            String[] splitted = weightsList.get(i).split(":");
            int layer = Integer.parseInt(splitted[0]);
            int start = Integer.parseInt(splitted[1]);
            int end = Integer.parseInt(splitted[2]);
            if (start != -1) {
                Weight weight = new Weight(layers[layer][start], layers[layer + 1][end]);
                weight.setValueAndDelta(Float.parseFloat(splitted[3]), Float.parseFloat(splitted[4]));
                this.layers[layer][start].connectWithOut(weight);
                this.layers[layer + 1][end].connectWithIn(weight);
            }
            else if (withBias) {
                Weight weight = new Weight(biases[layer], layers[layer + 1][end]);
                weight.setValueAndDelta(Float.parseFloat(splitted[3]), Float.parseFloat(splitted[4]));
                this.biases[layer].connectWithOut(weight);
                this.layers[layer + 1][end].connectWithIn(weight);
            }
        }
        Utils.delete(new File(directory));
    }

    public void setInputIn(float[] data) throws Exception {
        if (data.length != layers[0].length)
            throw new Exception("count of data != count of input neurons");

        for (int i = 0; i < layers[0].length; i++)
            layers[0][i].setValueIn(data[i]);

        for (int i = 1; i < layers.length; i++){
            for(int j = 0; j < layers[i].length; j++) {
                float sum = 0;
                for (Weight weight : layers[i][j].getConnectionsIn())
                    sum += weight.getStart().getValueOut() * weight.getValue();
                layers[i][j].setValueIn(sum);
            }
        }
    }

    public float[] getResult(){
        Neuron[] lastLayer = layers[layers.length-1];
        float[] result = new float[lastLayer.length];
        for (int i = 0; i < lastLayer.length; i++)
            result[i] = lastLayer[i].getValueIn();
        return result;
    }

    public float[] correctWeight(float[] requiredResult) throws Exception{
        Neuron[] lastLayer = layers[layers.length-1];
        if (requiredResult.length != lastLayer.length)
            throw new Exception("count of requiredResult != count of out neurons");

        error += getMeanSquaredError(requiredResult);
        iterations++;

        for (int i = 0; i < lastLayer.length; i++)
            lastLayer[i].delta = (requiredResult[i] - lastLayer[i].getValueOut()) * lastLayer[i].getActivationDerivativeIn();

        for (int i = layers.length-2; i >= 0; i--)
            for (int j = 0; j < layers[i].length; j++) {
                float sum = 0;
                for (Weight weight : layers[i][j].getConnectionsOut())
                    sum += weight.getValue() * weight.getEnd().delta;
                layers[i][j].delta = sum * (layers[i][j].getActivation() == ActivationFunctions.SIGMOID ? (1 - layers[i][j].getValueOut()) * layers[i][j].getValueOut() : layers[i][j].getActivationDerivativeIn());
            }

        for(int i = 0; i < layers.length - 1; i++)
            for(int j = 0; j < layers[i].length; j++){
                for (Weight weight : layers[i][j].getConnectionsOut())
                    weight.delta(learnSpeed * (weight.getStart().getValueOut() * weight.getEnd().delta) + momentum * weight.getLastDelta());
            }

        if (withBias)
            for (int i = 0; i < biases.length; i++)
                for (Weight weight : biases[i].getConnectionsOut())
                    weight.delta(learnSpeed * (weight.getStart().getValueOut() * weight.getEnd().delta) + momentum * weight.getLastDelta());

        float[] deltas = new float[layers[0].length];
        for (int i = 0; i < layers[0].length; i++)
            deltas[i] = layers[0][i].delta;
        return deltas;
    }

    public float iterateEra(){
        float error = this.error;
        int iterations = this.iterations;
        this.error = 0;
        this.iterations = 0;
        return error/(float)iterations;
    }

    public float getMeanSquaredError(float[] requiredResult) throws Exception{
        Neuron[] lastLayer = layers[layers.length-1];
        if (requiredResult.length != lastLayer.length)
            throw new Exception("count of requiredResult != count of out neurons");

        float result = 0;
        for (int i = 0; i < lastLayer.length; i++) {
            result += Math.pow(requiredResult[i] - lastLayer[i].getValueOut(), 2);
        }
        return result / (float)lastLayer.length;
    }

    public void save(String name) throws IOException {
        if (!Files.exists(Paths.get("./saves")))
            Files.createDirectory(Paths.get("./saves"));

        String directory = "./saves/" + name;
        File saveFile = new File( directory + ".fn");
        if (saveFile.exists())
            saveFile.delete();

        Files.createDirectory(Paths.get(directory));

        List<String> infoList = new ArrayList<>();
        infoList.add(learnSpeed+"");
        infoList.add(momentum+"");
        infoList.add(withBias+"");
        for (int i = 0; i < layers.length; i++)
            infoList.add(layers[i].length+"");
        Files.write(Paths.get(directory + "/info"), infoList);

        List<String> weightsList = new ArrayList<>();
        for (int i = 0; i < layers.length-1; i++){
            for (int j = 0; j < layers[i].length; j++){
                for (Weight weight : layers[i][j].getConnectionsOut())
                    weightsList.add(String.join(":", weight.getData()));
            }
        }
        for (int i = 0; i < biases.length; i++)
            for (Weight weight : biases[i].getConnectionsOut())
                weightsList.add(String.join(":", weight.getData()));

        Files.write(Paths.get(directory+ "/weights"), weightsList);

        try(ZipOutputStream zout = new ZipOutputStream(new FileOutputStream(directory + ".fn"));
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

    public float[] getInput(float[] output/*, int seed*/){
        //Random rnd = seed == -1 ? new Random() : new Random(seed);
        float[] prevLayer = output;
        for (int i = layers.length - 1; i >= 1; i--){
            float[] cofs = new float[prevLayer.length];
            float[][] weightMatrix = new float[prevLayer.length][];
            for (int j = 0; j < prevLayer.length; j++){
                cofs[j] = layers[i][j].getActivation().inverseFunc(prevLayer[j]);
                List<Weight> weights = layers[i][j].getConnectionsIn();
                weights.removeIf(x -> x.getStart().getIndex() == -1);
                float[] mass = new float[weights.size()];
                for (int m = 0; m < weights.size(); m++)
                    mass[m] = weights.get(m).getValue();
                weightMatrix[j] = mass;
            }
            prevLayer = Utils.pickInputs(cofs, weightMatrix, 10000);
        }
        return prevLayer;
    }
}
