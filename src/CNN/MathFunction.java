package CNN;

public class MathFunction {
	
	public static double sigmoid(double input){
		return (1 / (1 + Math.exp(-input)));
	}
	
	public static double ReLU(double input){
		if(input > 0) return input;
		else return 0;
	}
	
	public static double ReLUGrandient(double input){
		if(input > 0) return 1;
		else return 0;
	}
	
	/**
	 * 解释：此方法用于将几个Map转化成一串连续的一维向量
	 * @param maps
	 * @return
	 */
	public static double[] mapToLine(Maps[] maps) {
		int length = maps.length;
		int size = maps[0].getHeight();
		int returnSize = length * size * size;
		double[] returnLine = new double[returnSize];
		for(int i = 0;i < length;i++){
			for(int j = 0;j < size;j++){
				for(int k = 0;k < size;k++){
					returnLine[i * size * size + j * size + k] = maps[i].getNumber(j, k);
				}
			}
		}
		return returnLine;
	}
 	/**
	 * 解释 :本方法用于计算【原矩阵】中的某一个点经过卷积以后的结构，如果卷积核为5 * 5，那么计算范围
	 *      包括该点以及该点方圆两个
	 * @param originMap 原来的矩阵
	 * @param row 要算的点在原来的矩阵中哪一行
	 * @param column 要算的点在原来的矩阵中哪一列
	 * @param ck 卷积核对象
	 * @return 返回计算结果
	 */
	public static double calculateConvolutionalPoint(Maps originMap,int row,int column,ConvolutionalKernel ck){
		double[][] matrix = originMap.getMatrix();
		double temp = 0;
		int kernelSize = ck.getWidth();
		int back = (kernelSize - 1) / 2;
		for(int i = 0;i < kernelSize;i++){
			for(int j = 0;j < kernelSize;j++){
				temp += matrix[row-back+i][column-back+j] * ck.getWeight(i, j);
			}
		}
		temp += temp + ck.getBias();
		return temp;
	}
	
	public static double calculateSubsamplePoint(Maps originMap,int row,int column,SubsampleKernel sk){
		double[][] matrix = originMap.getMatrix();
		double temp = 0;
		temp = (matrix[row][column] + matrix[row][column+1] + matrix[row+1][column] + matrix[row+1][column+1])/4;
		//temp  = temp * sk.getBeta() + sk.getBias();
		return temp;
	}
}
