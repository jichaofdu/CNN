package CNN;

import java.io.IOException;

public class Launcher {
	public static void main(String[] args){
		CNNClassification cnn = new CNNClassification();
		try {
			cnn.trainingProcedure();
			cnn.testingProcedure();
		} catch (IOException | ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
