package ru.minebot.Forward;

import ru.minebot.ActivationFunctions;

import java.util.List;
import java.util.function.Function;

// TODO
public class Neuron {

    public float delta;

    public Neuron(int layer, int index, ActivationFunctions.ActivationFunction activationFunction){

    }

    public void setLayer(int i) {

    }

    public void connectWithOut(Weight weight) {

    }

    public void connectWithIn(Weight weight) {

    }

    public void setValueIn(float i) {

    }

    public List<Weight> getConnectionsIn() {

    }

    public float getValueIn() {
        return 0;
    }

    public float getValueOut() {
        return 0;
    }

    public float getActivationDerivativeIn() {
        return 0;
    }

    public List<Weight> getConnectionsOut() {

    }

    public ActivationFunctions.ActivationFunction getActivation() {

    }

    public int getIndex() {
        return 0;
    }
}