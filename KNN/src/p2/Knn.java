package p2;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Scanner;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.estimators.KDConditionalEstimator;

public class Knn {
	private List<KnnData> train_data_set;
	private List<KnnData> test_data_set;
	private int k;
	
	
	public Knn(List<KnnData> train_data_set, List<KnnData> test_data_set, int k) {
		//super();
		this.train_data_set = train_data_set;
		this.test_data_set = test_data_set;
		this.k = k;
	}
	
	/**
	 * read data from file
	 * @param filename
	 * @return
	 * @throws IOException
	 */
	public static List<KnnData> readDataFile(String filename) throws IOException {
		BufferedReader inputReader = null;
 
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
		Instances read_data = new Instances(inputReader);
		
		List<KnnData> list_data = new ArrayList<KnnData>();
		for(int i = 0; i < read_data.numInstances(); i ++){
			List<Double> attributes = new ArrayList<Double>();
			Instance current_instance = read_data.instance(i);
			int num_attributes = current_instance.numAttributes() ;
			int classifier = (int) current_instance.value(num_attributes - 1);
			for(int j = 0; j < num_attributes - 1; j++){
				attributes.add(current_instance.value(j));
			}
			KnnData knn_data = new KnnData(attributes, classifier); 
			list_data.add(knn_data);
		}
		return list_data;
	}

	/**
	 * calculate distance between two data
	 * @param k1
	 * @param k2
	 * @return
	 */
	public static Double calDistance(KnnData k1, KnnData k2){
		List<Double> k1_data = k1.getData();
		List<Double> k2_data = k2.getData();
		double distance = 0;
		if(k1_data.size() == k2_data.size()){
			for (int i = 0; i < k1_data.size(); i++){
				distance += Math.pow(k1_data.get(i), k2_data.get(i));
			}
			distance = Math.sqrt(distance);
		}
		return distance;
	}
	
	
	public List<KnnData> getKNearestNeighbors(KnnData test, int k){
		Comparator<KnnData> comparator = new DistanceComparator();
		int initial_size = train_data_set.size();
	    PriorityQueue<KnnData> queue = new PriorityQueue<KnnData>(initial_size, comparator);
		List<KnnData> k_nearest_neighbors = new ArrayList<KnnData>();
		for(int i = 0; i < train_data_set.size(); i++){
			KnnData current_train_data = train_data_set.get(i);
			double dis = calDistance(test, current_train_data);
			current_train_data.setDistance(dis);
			queue.add(current_train_data);
		}
		int i = 0;
		while(i < k){
			k_nearest_neighbors.add(queue.remove());
			i++;
		}
		/*for(KnnData kd : k_nearest_neighbors){
			System.out.println(kd.getClassifier()+"-=-"+kd.getDistance());
		}*/
		return k_nearest_neighbors;
	}
	
	public int determineClass(List<KnnData> k_nearest_neighbors){
		int counter0 = 0;
		int counter1 = 0;
		for(KnnData kd : k_nearest_neighbors){
			int kd_class = kd.getClassifier();
			if(kd_class == 0){
				counter0++;
			}
			if(kd_class == 1){
				counter1++;
			}
		}
		if(counter0 >= counter1){
			return 0;
		}else{
			return 1;
		}
		
	}
	/**
	 * 
	 * @param randomKd
	 * @param hitOrMiss 0:hit, 1:miss
	 * @return
	 */
	public KnnData findNearestX(KnnData randomKd, int hitOrMiss){
		KnnData result = null;
		if(hitOrMiss == 0){
			result = findNearestXHit(randomKd);
			//System.out.println(randomKd.getClassifier() + "hit "+ result.getClassifier() + ":" + result.getDistance());
		}else{
			result = findNearestXMiss(randomKd);
			//System.out.println(randomKd.getClassifier() + "miss "+ result.getClassifier() + ":" + result.getDistance());
		}
		return result;
	}
	
	public KnnData findNearestXHit(KnnData randomKd){
		List<KnnData> all_neighbors = getKNearestNeighbors(randomKd, train_data_set.size());
		KnnData result = null;
		for(KnnData kd : all_neighbors){
			if(kd.getClassifier() == randomKd.getClassifier()){
				result = kd;
				break;
			}
		}
		return result;
	}

	public KnnData findNearestXMiss(KnnData randomKd){
		List<KnnData> all_neighbors = getKNearestNeighbors(randomKd, train_data_set.size());
		KnnData result = null;
		for(KnnData kd : all_neighbors){
			if(kd.getClassifier() != randomKd.getClassifier()){
				result = kd;
				break;
			}
		}
		return result;
	}	
	public double[] assignWeights(){
		Random random = new Random();
		int index = random.nextInt(train_data_set.size());
		KnnData randomKd = train_data_set.get(index);
		double[] weights = new double[randomKd.getData().size()];
		//HashMap<Integer, Double> weights = new HashMap<Integer, Double>();
		//double weight = 0;
		KnnData xHit = findNearestX(randomKd, 0);
		KnnData xMiss = findNearestX(randomKd, 1);
		for (int i = 0; i < xHit.getData().size(); i++){
			weights[i] = weights[i] - Math.abs(xHit.getData().get(i) - randomKd.getData().get(i)) + Math.abs(xMiss.getData().get(i) - randomKd.getData().get(i));
			
		}
		return weights;
	}
	
	
	public int[] selesctFeatures(double[] weights){
		
		List<Double> weights_list = new ArrayList<Double>();
		for(int i = 0; i < weights.length; i++){
			weights_list.add(weights[i]);
		}
		
		int [] features_indexes = new int[14];
		Arrays.sort(weights);
		
		int weights_index = weights.length - 1;
		for(int i = 0; i < 14; i++){
			int index = weights_list.indexOf(weights[weights_index]);
			//System.out.println(weights[weights_index] + "index:" + index);
			features_indexes[i] = index;
			weights_index--;
		}
		return features_indexes;
	}
	
	public static Double calFeaturedDistance(KnnData k1, KnnData k2, int[] features_indexes){
		List<Double> k1_data = k1.getData();
		List<Double> k2_data = k2.getData();
		double distance = 0;
		if(k1_data.size() == k2_data.size()){
			for (int i = 0; i < features_indexes.length; i++){
				int current_index = features_indexes[i];
				distance += Math.pow(k1_data.get(current_index), k2_data.get(current_index));
			}
			distance = Math.sqrt(distance);
		}
		return distance;
	}
	
	public List<KnnData> getKFeatureNearestNeighbors(KnnData test, int k, int[] features_indexes){
		Comparator<KnnData> comparator = new DistanceComparator();
		int initial_size = train_data_set.size();
	    PriorityQueue<KnnData> queue = new PriorityQueue<KnnData>(initial_size, comparator);
		List<KnnData> k_nearest_neighbors = new ArrayList<KnnData>();
		for(int i = 0; i < train_data_set.size(); i++){
			KnnData current_train_data = train_data_set.get(i);
			double dis = calFeaturedDistance(test, current_train_data, features_indexes);
			current_train_data.setDistance(dis);
			queue.add(current_train_data);
		}
		int i = 0;
		while(i < k){
			k_nearest_neighbors.add(queue.remove());
			i++;
		}
		return k_nearest_neighbors;
	}
	
	public static Double calWeightedDistance(KnnData k1, KnnData k2, double[] weights){
		List<Double> k1_data = k1.getData();
		List<Double> k2_data = k2.getData();
		double distance = 0;
		if(k1_data.size() == k2_data.size()){
			for (int i = 0; i < k1_data.size(); i++){
				distance += weights[i] * Math.pow(k1_data.get(i), k2_data.get(i));
			}
			distance = Math.sqrt(distance);
		}
		return distance;
	}
	public List<KnnData> getKWeightedNearestNeighbors(KnnData test, int k, double[] weights){
		Comparator<KnnData> comparator = new DistanceComparator();
		int initial_size = train_data_set.size();
	    PriorityQueue<KnnData> queue = new PriorityQueue<KnnData>(initial_size, comparator);
		List<KnnData> k_nearest_neighbors = new ArrayList<KnnData>();
		for(int i = 0; i < train_data_set.size(); i++){
			KnnData current_train_data = train_data_set.get(i);
			double dis = calWeightedDistance(test, current_train_data, weights);
			current_train_data.setDistance(dis);
			queue.add(current_train_data);
		}
		int i = 0;
		while(i < k){
			k_nearest_neighbors.add(queue.remove());
			i++;
		}
		return k_nearest_neighbors;
	}
	
	
	
	public static void main(String[] args) throws Exception {
		int [] features_sizes = new int [] {14, 24, 34, 44, 54, 64, 74, 84, 94};
		for (int i = 0; i < features_sizes.length; i++){
			List<KnnData> train_data = readDataFile("data/" + features_sizes[i] + "_train_norm.arff");
			List<KnnData> test_data = readDataFile("data/" + features_sizes[i] + "_test_norm.arff");
			int k = 1;
			Knn knn1 = new Knn(train_data, test_data, k);
			double total_tests = test_data.size();
			double accurate_results0 = 0;
			double accurate_results1 = 0;
			double accurate_results2 = 0;
			for (KnnData kd : knn1.getTest_data_set()){
				int kd_classifier = kd.getClassifier();
				List<KnnData> k_nearest_neighbors0 = knn1.getKNearestNeighbors(kd, knn1.getK());
				
				double [] weights= knn1.assignWeights();
				List<KnnData> k_nearest_neighbors1 = knn1.getKWeightedNearestNeighbors(kd, knn1.getK(), weights);
				
				int[] features_indexes = knn1.selesctFeatures(weights);
				List<KnnData> k_nearest_neighbors2 = knn1.getKFeatureNearestNeighbors(kd, knn1.getK(), features_indexes);
				
				
				int classifier0 = knn1.determineClass(k_nearest_neighbors0);
				int classifier1 = knn1.determineClass(k_nearest_neighbors1);
				int classifier2 = knn1.determineClass(k_nearest_neighbors2);
				if(classifier0 == kd_classifier){
					accurate_results0++;
				}
				
				if(classifier1 == kd_classifier){
					accurate_results1++;
				}
				
				if(classifier2 == kd_classifier){
					accurate_results2++;
				}
			}
			double accuracy0 = (accurate_results0 / total_tests);
			System.out.println(features_sizes[i] + " " + accuracy0);
			//System.out.println("---------weighted results------------");
			double accuracy1 = (accurate_results1 / total_tests);
			System.out.println("weights: " + features_sizes[i] + " " + accuracy1);
			//System.out.println("---------fearure results------------");
			double accuracy2 = (accurate_results2 / total_tests);
			System.out.println("features: " + features_sizes[i] + " " + accuracy2);
			System.out.println("**************************************");
			
			
		}
		
	}
	public List<KnnData> getTrain_data_set() {
		return train_data_set;
	}
	public void setTrain_data_set(List<KnnData> train_data_set) {
		this.train_data_set = train_data_set;
	}
	public List<KnnData> getTest_data_set() {
		return test_data_set;
	}
	public void setTest_data_set(List<KnnData> test_data_set) {
		this.test_data_set = test_data_set;
	}
	public int getK() {
		return k;
	}
	public void setK(int k) {
		this.k = k;
	}
}
