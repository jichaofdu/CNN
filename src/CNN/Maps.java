package CNN;

public class Maps {
	private int width;
	private int height;
	private double[][] matrix;
	private double[][] rawValueMatrix;
	private double[][] error;
	
	public Maps(int width){
		this.width = width;
		this.height = width;
		this.matrix = new double[height][width];
		this.error = new double[height][width];
		this.rawValueMatrix = new double[height][width];
		for(int i = 0;i < height;i++){
			for(int j = 0;j < width;j++){
				this.matrix[i][j] = 0;
				this.error[i][j] = 0;
				this.rawValueMatrix[i][j] = 0;
			}
		}
	}
	
	public Maps(int width,double[][] matrix){
		this.width = width;
		this.height = width;
		matrix = new double[height][width];
		error = new double[height][width];
		for(int i = 0;i < height;i++){
			for(int j = 0;j < width;j++){
				this.matrix[i][j] = matrix[i][j];
				this.error[i][j] = 0;
			}
		}
	}
	
	public double getError(int i,int j){
		return error[i][j];
	}
	
	public void setError(int i,int j,double value){
		error[i][j] = value;
	}
	
	public double[][] getMatrix(){
		return this.matrix;
	}
	
	public int getWidth(){
		return this.width;
	}
	
	public int getHeight(){
		return this.height;
	}
	
	public double getNumber(int i,int j){
		return this.matrix[i][j];
	}
	
	public void setNumber(int i,int j,double number){
		this.matrix[i][j] = number;
	}
	
	public void setRawValue(int i,int j,double value){
		this.rawValueMatrix[i][j] = value;
	}
	
	public double getRawValue(int i,int j){
		return this.rawValueMatrix[i][j];
	}
	
	public void resetChange(){
		for(int i = 0;i < width;i++){
			for(int j = 0;j <width;j++){
				this.error[i][j] = 0;
			}
		}
	}
	
}
