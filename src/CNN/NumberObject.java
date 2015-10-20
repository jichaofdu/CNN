package CNN;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

public class NumberObject {
	private int height;
	private int width;
	private String path;
	private double[][] numberMatrix;
	private int actualNumber;
    private double meanValue;
    private double standardDeviation;
	
	public NumberObject(int indexOfFile,int height,int width,String path){
		this.height = height;
		this.width = width;
		this.path = path;
		try {
			readRawNumber(indexOfFile);
			calculateMeanAndStandardDeviation();		
			normalizeNumber();  	
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public NumberObject(String fileName){
		try {
			readRawNumberByName(fileName);
			calculateMeanAndStandardDeviation();		
			normalizeNumber();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private void readRawNumber(int indexOfFile) throws IOException{
		this.numberMatrix = new double[height][width];
		File file = new File(path + indexOfFile + ".txt");
		if(file.isFile() && file.exists()){
             InputStreamReader read = new InputStreamReader(new FileInputStream(file));
             BufferedReader bufferedReader = new BufferedReader(read);
             String lineTxt = null;
             for(int i = 0;i < height;i++){
            	 lineTxt = bufferedReader.readLine();
            	 String[] tempList = lineTxt.split(" ");
            	 for(int j = 0;j < width;j++){
            		 this.numberMatrix[i][j] = Integer.parseInt(tempList[j]);
            	 }
             }
             try{
            	 this.actualNumber = Integer.parseInt(bufferedReader.readLine());
            	
             }catch(Exception e){
            	 this.actualNumber = -1;
             }        
             read.close();
		}
		double[][] numberMatrix2 = new double[32][32];
		for(int i = 0;i < 32;i++){
			for(int j = 0;j < 32;j++){
				numberMatrix2[i][j] = 0;
			}
		}
		for(int i = 0;i < 28;i++){
			for(int j = 0;j < 28;j++){
				numberMatrix2[i+2][j+2] = this.numberMatrix[i][j];
			}
		}
		this.height = 32;
		this.width = 32;
		this.numberMatrix = numberMatrix2;
	}

	private void readRawNumberByName(String fileName) throws IOException{
		numberMatrix = new double[height][width];
		File file = new File(path + fileName);
		if(file.isFile() && file.exists()){
            InputStreamReader read = new InputStreamReader(new FileInputStream(file));
            BufferedReader bufferedReader = new BufferedReader(read);
            String lineTxt = null;
            for(int i = 0;i < height;i++){
           	 lineTxt = bufferedReader.readLine();
           	 String[] tempList = lineTxt.split(" ");
           	 for(int j = 0;j < width;j++){
           		 numberMatrix[i][j] = Integer.parseInt(tempList[j]);
           	 }
            }
            try{
           	 actualNumber = Integer.parseInt(bufferedReader.readLine());
            }catch(Exception e){
           	 actualNumber = -1;
            }        
            read.close();
		}
	}
	
	private void calculateMeanAndStandardDeviation(){
		double amount = 0;
        for(int i = 0;i < height;i++){
        	for(int j = 0;j < width;j++){
       			amount += numberMatrix[i][j];
       	 	}
        }
        meanValue = amount / (height * width);
        amount = 0;
        for(int i = 0;i < height;i++){
        	for(int j = 0;j < width;j++){
       			amount += Math.pow(numberMatrix[i][j] - meanValue, 2);
       	 	}
        }
        standardDeviation = Math.sqrt(amount / (height * width));
	}
	
	private void normalizeNumber(){
		for(int i = 0;i < height;i++){
			for(int j = 0;j < width;j++){
				numberMatrix[i][j] = (numberMatrix[i][j] - meanValue) / standardDeviation;
			}
		}
	}
	
	public double getValue(int y,int x){
		return numberMatrix[y][x];
	}
	
	public int getActualNumber() {
		return actualNumber;
	}	
}
