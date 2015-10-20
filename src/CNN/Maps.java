package CNN;

public class Maps {
	private int width;
	private int height;
	private double[][] matrix;
	
	public Maps(int width){
		this.width = width;
		this.height = width;
		matrix = new double[height][width];
		for(int i = 0;i < height;i++){
			for(int j = 0;j < width;j++){
				this.matrix[i][j] = 0;
			}
		}
	}
	
	public Maps(int width,double[][] matrix){
		this.width = width;
		this.height = width;
		matrix = new double[height][width];
		for(int i = 0;i < height;i++){
			for(int j = 0;j < width;j++){
				this.matrix[i][j] = matrix[i][j];
			}
		}
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
}
