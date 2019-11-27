package global.skymind.training.facial_recognition.identification;

import global.skymind.training.facial_recognition.detection.FaceLocalization;

public class Prediction {

    private String label;
    private double score;
    private FaceLocalization faceLocalization;

    public Prediction(String label, double score, FaceLocalization faceLocalization) {
        this.label = label;
        this.score = score;
        this.faceLocalization = faceLocalization;
    }

    public Prediction(String label, double percentage) {
        this.label = label;
        this.score = percentage;
    }

    public String getLabel(){
        return this.label;
    }

    public double getScore(){
        return this.score;
    }

    public FaceLocalization getFaceLocalization(){
        return this.faceLocalization;
    }

    public String toString() {
//        return String.format("%s: %.2f ", this.label, this.score);
        return String.format("%s", this.label);
    }
}