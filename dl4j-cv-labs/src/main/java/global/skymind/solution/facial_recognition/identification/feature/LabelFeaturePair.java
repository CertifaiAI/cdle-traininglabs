package global.skymind.solution.facial_recognition.identification.feature;

import org.nd4j.linalg.api.ndarray.INDArray;

public class LabelFeaturePair {
    private final String label;
    private final INDArray embedding;

    public LabelFeaturePair(String label, INDArray embedding) {
        this.label = label;
        this.embedding = embedding;
    }

    public INDArray getEmbedding() {
        return this.embedding;
    }

    public String getLabel() {
        return this.label;
    }
}