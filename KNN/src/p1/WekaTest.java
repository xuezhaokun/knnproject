package p1;
import java.io.BufferedReader;
import java.io.FileReader;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

//Reference: Based on TA's instruction and example on Piazza
/**
 * Calculate results for J48
 * @author Zhaokun Xue
 *
 */
public class WekaTest {
 
	public static void main(String[] args) throws Exception {
		int [] features_sizes = new int [] {14, 24, 34, 44, 54, 64, 74, 84, 94};
		// go through each dataset
		for (int i = 0; i < features_sizes.length; i++){
			//read train data
			BufferedReader train_data_file = new BufferedReader(new FileReader("data/"+features_sizes[i]+"_train_norm.arff"));
			// read test data
			BufferedReader test_data_file = new BufferedReader(new FileReader("data/"+features_sizes[i]+"_test_norm.arff"));
			
			//construct instances using weka
			Instances train = new Instances(train_data_file);
	 		Instances test =  new Instances(test_data_file);
	 		
	 		//choose the first (n -1) attributes as data features and the last one as result
	 		train.setClassIndex(train.numAttributes() - 1);
	 		test.setClassIndex(test.numAttributes() - 1);
	 		
	 		//classify by J48
	 		Classifier cls = new J48();
	 		cls.buildClassifier(train);
	 		
	 		//evaluation
	 		Evaluation eval = new Evaluation(train);
	 		eval.evaluateModel(cls, test);
	 		
	 		//calculate accuracy and print out results 
	 		double accuracy = eval.pctCorrect()/100;
	 		System.out.print(features_sizes[i] + " " + accuracy + "\n");      
		}

 	}
}