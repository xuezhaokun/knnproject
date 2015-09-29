package p2;

import java.util.List;
/**
 * KNN data for each record of data
 * @author Zhaokun Xue
 *
 */
public class KnnData {
	private List<Double> data;
	private int classifier;
	private double distance; // the distance from test Knn Data
	
	/**
	 * constructor for KnnData
	 * @param data the record's data
	 * @param classifier the class of the data record
	 */
	public KnnData(List<Double> data, int classifier) {
		this.data = data;
		this.classifier = classifier;
		this.distance = Double.POSITIVE_INFINITY;
	}
	
	// getters and setters
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
