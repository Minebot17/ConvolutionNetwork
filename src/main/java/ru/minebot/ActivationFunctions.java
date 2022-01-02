package ru.minebot;

import java.util.function.Function;

public class ActivationFunctions {

    // TODO fill functions
    public static ActivationFunction LINEAR = null;
    public static ActivationFunction RELU = null;
    public static ActivationFunction SIGMOID = null;

    public class ActivationFunction {

        private final Function<Float, Float> function;
        private final Function<Float, Float> inverseFunction;

        public ActivationFunction(Function<Float, Float> function, Function<Float, Float> inverseFunction){
            this.function = function;
            this.inverseFunction = inverseFunction;
        }

        public float invoke(float input){
            return function.apply(input);
        }

        public float inverseFunc(float input){
            return inverseFunction.apply(input);
        }
    }
}