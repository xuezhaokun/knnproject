package p2;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
/**
 * Implement KNN Method
 * @author Zhaokun Xue
 *
 */
public class Knn {
	private List<KnnData> train_data_set;
	private List<KnnData> test_data_set;
	private int k;
	
	/**
	 * constructor for KNN
	 * @param train_data_set train data set
	 * @param test_data_set	test data set
	 * @param k k value
	 */
	public Knn(List<KnnData> train_data_set, List<KnnData> test_data_set, int k) {
		this.train_data_set = train_data_set;
		this.test_data_set = test_data_set;
		this.k = k;
	}
	
	/**
	 * read data from file using weka instance feature
	 * @param filename input file name
	 * @return a list of KnnData
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
	 * calculate distance between two Knndata
	 * @param k1
	 * @param k2
	 * @return distance between k1 and k2
	 */
	public static double calDistance(KnnData k1, KnnData k2){
		List<Double> k1_data = k1.getData();
		List<Double> k2_data = k2.getData();
		double distance = 0;
		if(k1_data.size() == k2_data.size()){
			for (int i = 0; i < k1_data.size(); i++){
				double diff = k1_data.get(i) - k2_data.get(i);
				distance += Math.pow(diff, 2);
			}
			distance = Math.sqrt(distance);
		}
		return distance;
	}
	
	/**
	 * get the k nearest neighbors for test example
	 * @param test the test example
	 * @param k value k
	 * @return a list of KnnData contains k nearest neighbors
	 */
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
		return k_nearest_neighbors;
	}
	
	/**
	 * determine the class based on the test data's k nearest neighbors
	 * @param k_nearest_neighbors the test data's k nearest neighbors
	 * @return 0 or 1 for the class
	 */
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
	
	/***************  methods for relief improvements ********************/	
	
	/**
	 * find the nearest neighbor with the same class
	 * @param randomKd a random input KnnData
	 * @return the nearest neighbor with the same class
	 */
	public KnnData findNearestXHit(KnnData randomKd){
		KnnData result = null;
		List<KnnData> all_neighbors = getKNearestNeighbors(randomKd, train_data_set.size());
		for(KnnData kd : all_neighbors){
			if(kd.getClassifier() == randomKd.getClassifier()){
				result = kd;
				break;
			}
		}
		return result;
	}
	
	/**
	 * find the nearest neighbor with the different class
	 * @param randomKd a random input KnnData
	 * @return the nearest neighbor with the different class
	 */
	public KnnData findNearestXMiss(KnnData randomKd){
		KnnData result = null;
		List<KnnData> all_neighbors = getKNearestNeighbors(randomKd, train_data_set.size());
		for(KnnData kd : all_neighbors){
			if(kd.getClassifier() != randomKd.getClassifier()){
				result = kd;
				break;
			}
		}
		return result;
	}	

	/**
	 * assign weights to each features 
	 * @param m doing m times for updating weights
	 * @return an array containing the weights for corresponding features
	 */
	public double[] assignWeights(int m){
		double[] weights = new double[train_data_set.get(0).getData().size()];
		HashMap<KnnData, KnnData[]> hitMissMap = new HashMap<KnnData, KnnData[]>();
		
		for(int k = 0; k < m; k++){
			Random random = new Random();
			int index = random.nextInt(train_data_set.size());
			KnnData randomKd = train_data_set.get(index);
			KnnData xHit = null;
			KnnData xMiss = null;
			KnnData[] values = new KnnData[2];
			if(hitMissMap.containsKey(randomKd)){
				xHit = hitMissMap.get(randomKd)[0];
				xMiss = hitMissMap.get(randomKd)[1];
			}else{
				xHit =  findNearestXHit(randomKd);
				xMiss = findNearestXMiss(randomKd);
				values[0] = xHit;
				values[1] = xMiss;
				hitMissMap.put(randomKd, values);
			}
			int features_length = randomKd.getData().size();
			for(int i = 0; i < features_length; i++){
				weights[i] = weights[i] - Math.abs(xHit.getData().get(i) - randomKd.getData().get(i)) + Math.abs(xMiss.getData().get(i) - randomKd.getData().get(i));
			}
		}
		return weights;

	}

	/**
	 * find the selected features indexes
	 * @param weights the weights for corresponding features
	 * @return an array of indexes for the selected features
	 */
	public int[] selesctFeatures(double[] weights){
		List<Double> list_weights = new ArrayList<Double>();
		for(int i = 0; i < weights.length; i++){
			list_weights.add(weights[i]);
		}
		
		int[] features_indexes = new int[14];
		Arrays.sort(weights);
		int weights_index = weights.length - 1;

		for(int j = 0; j < 14; j++){
			int current_index = list_weights.indexOf(weights[weights_index]);
			features_indexes[j] = current_index;
			weights_index --;
		}

		return features_indexes; 
	}
	
	/**
	 * calculate the distance for feature selection relief
	 * @param k1 KnnData1
	 * @param k2 KnnData2
	 * @param features_indexes selected features' indexes
	 * @return the distance
	 */
	public static Double calFeaturedDistance(KnnData k1, KnnData k2, int[] features_indexes){
		List<Double> k1_data = k1.getData();
		List<Double> k2_data = k2.getData();
		double distance = 0;
		if(k1_data.size() == k2_data.size()){
			for (int i = 0; i < features_indexes.length; i++){
				int current_index = features_indexes[i];
				double diff = k1_data.get(current_index) - k2_data.get(current_index);
				distance +=  Math.pow(diff, 2);
			}
			distance = Math.sqrt(distance);
		}
		return distance;
	}
	
	/**
	 * get the k nearest neighbors for features selection improvements
	 * @param test test knnData
	 * @param k value k
	 * @param features_indexes selected features' indexes
	 * @return k nearest neighbors
	 */
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
	
	/**
	 * calculate distance for weighted distance relief improvement 
	 * @param k1 KnnData 1
	 * @param k2 KnnData 2
	 * @param weights weights for corresponding features
	 * @return the distance
	 */
	public static Double calWeightedDistance(KnnData k1, KnnData k2, double[] weights){
		List<Double> k1_data = k1.getData();
		List<Double> k2_data = k2.getData();
		double distance = 0;
		if(k1_data.size() == k2_data.size()){
			for (int i = 0; i < k1_data.size(); i++){
				double diff = k1_data.get(i) - k2_data.get(i);
				distance += weights[i] * Math.pow(diff, 2);
			}
			distance = Math.sqrt(distance);
		}
		return distance;
	}
	
	/**
	 * find the k nearest neighbors for weighted distance relief improvement
	 * @param test	test KnnData
	 * @param k value k
	 * @param weights weights for corresponding features
	 * @return k nearest neighbors
	 */
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
	
	// main method print out results for 3.3 and 3.4
	// need to manually change k and k94 for different results
	public static void main(String[] args) throws Exception {
		int [] features_sizes = new int [] {14, 24, 34, 44, 54, 64, 74, 84, 94};
		int k = 1;
		System.out.println("******Results for Relief*******" + "k=" + k);
		for (int i = 0; i < features_sizes.length; i++){
			List<KnnData> train_data = readDataFile("data/" + features_sizes[i] + "_train_norm.arff");
			List<KnnData> test_data = readDataFile("data/" + features_sizes[i] + "_test_norm.arff");
			
			Knn knn1 = new Knn(train_data, test_data, k);
			double total_tests = test_data.size();
			double accurate_results0 = 0;
			double accurate_results1 = 0;
			double accurate_results2 = 0;
			for (KnnData kd : knn1.getTest_data_set()){
				int kd_classifier = kd.getClassifier();
				List<KnnData> k_nearest_neighbors0 = knn1.getKNearestNeighbors(kd, knn1.getK());
				
				double[] weights= knn1.assignWeights(10000);

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
			System.out.println("pure: " + features_sizes[i] + " " + accuracy0);
			
			double accuracy1 = (accurate_results1 / total_tests);
			System.out.println("weights: " + features_sizes[i] + " " + accuracy1);
			
			double accuracy2 = (accurate_results2 / total_tests);
			System.out.println("features: " + features_sizes[i] + " " + accuracy2);
			System.out.println("-------------------------------");
		}
		int k94 = 1;
		System.out.println("******Results for Different m of 94 features******" + "k=" + k94);
		List<KnnData> train_data94 = readDataFile("data/94_train_norm.arff");
		List<KnnData> test_data94 = readDataFile("data/94_test_norm.arff");
		Knn knn94 = new Knn(train_data94, test_data94, k94);
		double total_tests94 = test_data94.size();
		int[] m = new int[]{100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
		for(int l = 0; l < m.length; l++){
			double accurate_results94_1 = 0;
			double accurate_results94_2 = 0;
			for(KnnData kd : test_data94){
				int kd_class = kd.getClassifier();
				double[] weights94= knn94.assignWeights(m[l]);
			
				List<KnnData> k_nearest_neighbors94_1 = knn94.getKWeightedNearestNeighbors(kd, knn94.getK(), weights94);
			
				int[] features_indexes94 = knn94.selesctFeatures(weights94);
				List<KnnData> k_nearest_neighbors94_2 = knn94.getKFeatureNearestNeighbors(kd, knn94.getK(), features_indexes94);
			
				int classifier94_1 = knn94.determineClass(k_nearest_neighbors94_1);
				int classifier94_2 = knn94.determineClass(k_nearest_neighbors94_2);
				if(classifier94_1 == kd_class){
					accurate_results94_1++;
				}
			
				if(classifier94_2 == kd_class){
					accurate_results94_2++;
				}
			}
			double accuracy94_1 = (accurate_results94_1 / total_tests94);
			System.out.println("weights: " + m[l] + " " + accuracy94_1);
			double accuracy94_2 = (accurate_results94_2 / total_tests94);
			System.out.println("features: " + m[l] + " " + accuracy94_2);
		
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
