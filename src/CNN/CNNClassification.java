package CNN;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Random;

public class CNNClassification {
	private Maps[] inputLayer;
	private Maps[] c1Layer;
	private Maps[] s2Layer;
	private Maps[] c3Layer;
	private Maps[] s4Layer;
	private Maps[] c5Layer;
	private Maps[] f6Layer;
	private Maps[] outputLayer;
	
	private ConvolutionalKernel[] ck1;
	private SubsampleKernel[] sk2;
	private ConvolutionalKernel[] ck3;
	private SubsampleKernel[] sk4;
	private GapParameter[] gap5;
	private IHFormulaParameter[] bp1;
	private HOFormulaParameter[] bp2;
	
	private int learningTimes;
	private int trainingSetSize;
	private int testSetSize;
	private int runSize;
	private int ckSize;
	private int stride;
	
	private int desiredNumber;
	private double[] desiredOutput;
	private int guessNumber;
	
	private String weightSavePath = "d:\\PJ1\\part2\\data\\";
	private String dataPath = "D:\\dataset\\";
	private String resultPath = "d:\\Test_Set_Result.txt";
	private String logPath = "d:\\Log_Record.txt";
	
	public CNNClassification(){
		this.learningTimes = HyperParameter.LEARNINGTIMES;
		this.trainingSetSize = HyperParameter.TRAININGSETSIZE;
		this.testSetSize = HyperParameter.TESTSETSIZE;
		this.runSize = HyperParameter.RUNSETSIZE;
		this.ckSize = HyperParameter.CKSIZE;
		this.stride = HyperParameter.STRIDE;
		initPara();
		generateLayers();
	}	
	
	public void trainingProcedure() throws IOException{
		LogRecord.logRecord("[Tip] Initialize Layers Success.",logPath);
		initPara();
		LogRecord.logRecord("[Tip] Initialize Weight Success.",logPath);
		for(int times = 0;times < learningTimes;times++){
			//For every times of training.
			for(int i = 1;i <= trainingSetSize;i++){
				NumberObject numberObj = new NumberObject(i,28,28,dataPath);
				setNowCase(numberObj);
				calculateOutput();
				guessNumberAndSaveAnswer();
				backPropagation();
				if((double)(i % 100) == 0 && i >= 100){
					System.out.println("["+ i + "] Runing......");
				}
				if((double)(i % 10000) == 0 && i >= 10000){
					saveParaToDisk();
					LogRecord.logRecord("[" + i + "] Saved weight to the disk. ",logPath);
				}
			}
		}
		LogRecord.logRecord("[End] Learning Procedure End ",logPath);
	}
	
	private void initPara(){
		Random randomgen = new Random();
		
		desiredOutput = new double[10];
		
		//Initialize weight and bias between intput and C1
		double[][] ck1TempWeight = new double[ckSize][ckSize];
		double ck1TempBias;
		this.ck1 = new ConvolutionalKernel[6];
		for(int i = 0;i < 6;i++){
			for(int j = 0;j < ckSize;j++){
				for(int k = 0;k < ckSize;k++){
					ck1TempWeight[j][k] = ((randomgen.nextDouble() - 0.5) * 2) / (ckSize * ckSize);
				}
			}
			ck1TempBias = (randomgen.nextDouble() - 0.5) * 2 * 0.2;
			ck1[i] = new ConvolutionalKernel(ckSize,ck1TempWeight,ck1TempBias,1,"ck1"+i);
		}
		//Initialize weight and bias between C1 and S2
		double sk2Weight;
		double sk2Bias;
		this.sk2 = new SubsampleKernel[6];
		for(int i = 0;i < 6;i++){
			sk2Weight = (randomgen.nextDouble() - 0.5) * 2;
			sk2Bias = (randomgen.nextDouble() - 0.5) * 2;
			sk2[i] = new SubsampleKernel(sk2Weight,sk2Bias,2,"sk2" + i);
		}
		//Initialize weight and bias between S2 and C3
		double[][] ck3TempWeight = new double[ckSize][ckSize];
		double ck3TempBias;
		this.ck3 = new ConvolutionalKernel[16];
		for(int i = 0;i < 16;i++){
			for(int j = 0;j < ckSize;j++){
				for(int k = 0;k < ckSize;k++){
					ck3TempWeight[j][k] = ((randomgen.nextDouble() - 0.5) * 2) / (ckSize * ckSize);
				}
			}
			ck3TempBias = (randomgen.nextDouble() - 0.5) * 2 * 0.2;
			ck3[i] = new ConvolutionalKernel(5,ck3TempWeight,ck3TempBias,1,"ck3"+i);
		}
		
		//Initialize weight and bias between C3 and S4
		double sk4Weight;
		double sk4Bias;
		this.sk4 = new SubsampleKernel[16];
		for(int i = 0;i < 16;i++){
			sk4Weight = (randomgen.nextDouble() - 0.5) * 2 * 0.2;
			sk4Bias = (randomgen.nextDouble() - 0.5) * 2 * 0.2;
			sk4[i] = new SubsampleKernel(sk4Weight,sk4Bias,2,"sk4" + i);
		}
		//Initialize weight and bias between S4 and C5
		double[] gapTempWeight = new double[16 * 5 * 5];
		double gapTempBias;
		this.gap5 = new GapParameter[120];
		for(int i = 0;i < 120;i++){
			for(int j = 0;j < 16 * 5 * 5;j++){
				gapTempWeight[j] = (randomgen.nextDouble() - 0.5) * 2 / Math.sqrt(16 * 5 * 5);
			}
			gapTempBias = randomgen.nextDouble() - 1.0d;
			gap5[i] = new GapParameter(16 * 5 * 5,gapTempWeight,gapTempBias,i);
		}
		//Initialize weight and bias between C5 and F6
		double[] bp1TempWeight = new double[120];
		double bp1TempBias;
		this.bp1 = new IHFormulaParameter[84];
		for(int i = 0;i < 84;i++){
			for(int j = 0;j < 120;j++){
				bp1TempWeight[j] = (randomgen.nextDouble() - 0.5) * 2 / Math.sqrt(120);
			}
			bp1TempBias = randomgen.nextDouble() - 1.0d;
			bp1[i] = new IHFormulaParameter(120,bp1TempWeight,bp1TempBias,i);
		}
		//Initialize weight and bias between F6 and output
		double[] bp2TempWeight = new double[84];
		double bp2TempBias;
		this.bp2 = new HOFormulaParameter[10];
		for(int i = 0;i < 10;i++){
			for(int j = 0;j < 84;j++){
				bp2TempWeight[j] = (randomgen.nextDouble() - 0.5) * 2 / Math.sqrt(84);
			}
			bp2TempBias = (randomgen.nextDouble() - 0.5) * 2 * 0.2;
			bp2[i] = new HOFormulaParameter(84,bp2TempWeight,bp2TempBias,i);
 		}
	}
	
	private void generateLayers(){
		inputLayer = new Maps[1];
		for(int i = 0;i < 1;i++){
			inputLayer[i] = new Maps(32);
		}
		c1Layer = new Maps[6];
		for(int i = 0;i < 6;i++){
			c1Layer[i] = new Maps(28);
		}
		s2Layer = new Maps[6];
		for(int i = 0;i < 6;i++){
			s2Layer[i] = new Maps(14);
		}
		c3Layer = new Maps[16];
		for(int i = 0;i < 16;i++){
			c3Layer[i] = new Maps(10);
		}
		s4Layer = new Maps[16];
		for(int i = 0;i < 16;i++){
			s4Layer[i] = new Maps(5);
		}
		c5Layer = new Maps[120];
		for(int i = 0;i < 120;i++){
			c5Layer[i] = new Maps(1);
		}
		f6Layer = new Maps[84];
		for(int i = 0;i < 84;i++){
			f6Layer[i] = new Maps(1);
		}
		outputLayer= new Maps[10];
		for(int i = 0;i < 10;i++){
			outputLayer[i] = new Maps(1);
		}
	}
	
	private void setNowCase(NumberObject nb){
		for(int i = 0;i < 32;i++){
			for(int j = 0;j < 32;j++){
				inputLayer[0].setNumber(i, j, nb.getValue(i, j));
			}
		}
		desiredNumber = nb.getActualNumber();
		for(int i = 0;i < 10;i++){
			desiredOutput[i] = 0;
		}
		if(desiredNumber >= 0){
			desiredOutput[desiredNumber] = 1;
		}
		
	}
	
	private void calculateOutput(){
		//首先计算第一层:卷积层
		double temp = 0;
		for(int i = 0;i < 6;i++){
			for(int j = 0;j < 28;j++){
				for(int k = 0;k < 28;k++){
					int j2 = j + 2;
					int k2 = k + 2;
					temp = MathFunction.calculateConvolutionalPoint(inputLayer[0],j2, k2, ck1[i]);
					c1Layer[i].setNumber(j, k, temp);
				}
			}
		}
		//计算第二层：子采样层
		for(int i = 0;i < 6;i++){
			for(int j = 0;j < 14;j++){
				for(int k = 0;k < 14;k++){
					int j2 = j * sk2[i].getStride();
					int k2 = k * sk2[i].getStride();
					temp = MathFunction.calculateSubsamplePoint(c1Layer[i],j2, k2, sk2[i]);
					s2Layer[i].setNumber(j, k, temp);
				}
			}
		}
		//计算第三层：卷积层
		for(int i = 0;i < 16;i++){
			for(int j = 0;j < 10;j++){
				for(int k = 0;k < 10;k++){
					int j2 = j + 2;
					int k2 = k + 2;
					temp = 0;
					for(int m = 0; m < 6;m++){
						temp += MathFunction.calculateConvolutionalPoint(s2Layer[m],j2, k2, ck3[i]);
					}
					c3Layer[i].setNumber(j, k, temp/6);
				}
			}
		}
		//计算第四层：子采样层
		for(int i = 0;i < 16;i++){
			for(int j = 0;j < 5;j++){
				for(int k = 0;k < 5;k++){
					int j2 = j * sk4[i].getStride();
					int k2 = k * sk4[i].getStride();
					temp = MathFunction.calculateSubsamplePoint(c3Layer[i],j2, k2, sk4[i]);
					s4Layer[i].setNumber(j, k, temp);
				}
			}
		}
		//计算gap层：将16 * 5 * 5个数字摊开，套用bp网络算法展开
		double[] gapLine = MathFunction.mapToLine(s4Layer);
		for(int i = 0;i < 120;i++){
			double tempHidden = 0;
			for(int j = 0;j < 16 * 5 * 5;j++){
				tempHidden += gapLine[j] * gap5[i].getWeight(j);
			}
			tempHidden = tempHidden + gap5[i].getBias();
			c5Layer[i].setNumber(0, 0, tempHidden);
		}
		//计算第六层：bp网络Hidden层
		for(int i = 0;i < 84;i++){
			double tempHidden = 0;
			for(int j = 0;j < 120;j++){
				tempHidden += bp1[i].getWeight(j) * c5Layer[j].getNumber(0, 0);
			}
			tempHidden = tempHidden + bp1[i].getBias();
			f6Layer[i].setNumber(0, 0, tempHidden); 
		}
		//计算第七层：output层
		for(int i = 0;i < 10;i++){
			double tempOutput = 0;
			for(int j = 0;j < 84;j++){
				tempOutput += bp2[i].getWeight(j) * f6Layer[j].getNumber(0, 0);
			}
			tempOutput = MathFunction.sigmoid(tempOutput + bp2[i].getBias());
			outputLayer[i].setNumber(0, 0, tempOutput); 
		}
	}
	
	private void guessNumberAndSaveAnswer() throws IOException{
		double max = 0;
		for(int i = 0;i < outputLayer.length;i++){
			if(outputLayer[i].getNumber(0, 0) > max){
				max = outputLayer[i].getNumber(0, 0);
				guessNumber = i;
			}
		}
		File file = new File(resultPath);
		BufferedWriter fw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file, true), "UTF-8"));
		fw.append("" + guessNumber);
		fw.newLine();
		fw.flush();
		fw.close();
	}
	
	private void backPropagation(){
		
		
	}
	
	private void readParaFromDisk() throws FileNotFoundException, ClassNotFoundException, IOException{
		for(int i = 0;i < 6;i++){
			ck1[i].readFromDiskCK(weightSavePath);
		}
		for(int i = 0; i < 6;i++){
			sk2[i].readFromDiskHO(weightSavePath);
		}
		for(int i = 0;i < 16;i++){
			ck3[i].readFromDiskCK(weightSavePath);
		}
		for(int i = 0;i < 16;i++){
			sk4[i].readFromDiskHO(weightSavePath);
		}
		for(int i = 0;i < 120;i++){
			gap5[i].readFromDiskGap(weightSavePath);
		}
		for(int i = 0;i < 84;i++){
			bp1[i].readFromDiskIH(weightSavePath);
		}
		for(int i = 0;i < 10;i++){
			bp2[i].readFromDiskHO(weightSavePath);
		}
	}
	
	private void saveParaToDisk() throws FileNotFoundException, IOException{
		for(int i = 0;i < 6;i++){
			ck1[i].writeToDiskCK(weightSavePath);
		}
		for(int i = 0; i < 6;i++){
			sk2[i].writeToDiskSK(weightSavePath);
		}
		for(int i = 0;i < 16;i++){
			ck3[i].writeToDiskCK(weightSavePath);
		}
		for(int i = 0;i < 16;i++){
			sk4[i].writeToDiskSK(weightSavePath);
		}
		for(int i = 0;i < 120;i++){
			gap5[i].writeToDiskGap(weightSavePath);
		}
		for(int i = 0;i < 84;i++){
			bp1[i].writeToDiskIH(weightSavePath);
		}
		for(int i = 0;i < 10;i++){
			bp2[i].writeToDiskHO(weightSavePath);
		}
	}
	
}
