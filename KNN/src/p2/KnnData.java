package p2;

import java.util.List;

public class KnnData {
	private List<Double> data;
	private int classifier;
	private double distance;
	
	public KnnData(List<Double> data, int classifier) {
		this.data = data;
		this.classifier = classifier;
		this.distance = Double.POSITIVE_INFINITY;
	}
	
	public List<Double> getData() {
		return data;
	}
	public void setData(List<Double> data) {
		this.data = data;
	}
	
	public int getClassifier() {
		return classifier;
	}
	public void setClassifier(int classifier) {
		this.classifier = classifier;
	}
	public double getDistance() {
		return distance;
	}
	public void setDistance(double distance) {
		this.distance = distance;
	}
}
