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
	private double[] gapLine;
	private double[] gapRawNumber;
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
	private double learningRate;
	
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
		this.learningRate = HyperParameter.learingRate;
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
				if((double)(i % 5000) == 0 && i >= 5000){
					saveParaToDisk();
					LogRecord.logRecord("[" + i + "] Saved weight to the disk. ",logPath);
				}
			}
		}
		LogRecord.logRecord("[End] Learning Procedure End ",logPath);
	}
	
	public void testingProcedure() throws FileNotFoundException, ClassNotFoundException, IOException{
		readParaFromDisk();
		int correct = 0;
		for(int i = 60000;i < 60001 + testSetSize;i++){
			NumberObject numberObj = new NumberObject(i,28,28,dataPath);
			setNowCase(numberObj);
			calculateOutput();
			guessNumberAndSaveAnswer();
			if(desiredNumber == guessNumber){
				correct++;
			}
			
		}
		LogRecord.logRecord("[End] Correct rate：" + correct + " / " + testSetSize,logPath);
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
		//Calculate NO.1 layer: c1 layer
		double temp = 0;
		for(int i = 0;i < 6;i++){
			for(int j = 0;j < 28;j++){
				for(int k = 0;k < 28;k++){
					int j2 = j + 2;
					int k2 = k + 2;
					temp = MathFunction.calculateConvolutionalPoint(inputLayer[0],j2, k2, ck1[i]);
					c1Layer[i].setRawValue(j, k, temp);
					c1Layer[i].setNumber(j, k, MathFunction.ReLU(temp));
				}
			}
		}
		//Calcualte NO.2 layer: s2 layer
		for(int i = 0;i < 6;i++){
			for(int j = 0;j < 14;j++){
				for(int k = 0;k < 14;k++){
					int j2 = j * sk2[i].getStride();
					int k2 = k * sk2[i].getStride();
					temp = MathFunction.calculateSubsamplePoint(c1Layer[i],j2, k2, sk2[i]);
					s2Layer[i].setRawValue(j, k, temp);
					s2Layer[i].setNumber(j, k, MathFunction.ReLU(temp));
				}
			}
		}
		//Calculate NO.3 layer: c3 layer
		for(int i = 0;i < 16;i++){
			for(int j = 0;j < 10;j++){
				for(int k = 0;k < 10;k++){
					int j2 = j + 2;
					int k2 = k + 2;
					temp = 0;
					for(int m = 0; m < 6;m++){
						temp += MathFunction.calculateConvolutionalPoint(s2Layer[m],j2, k2, ck3[i]);
					}
					c3Layer[i].setRawValue(j, k, temp / 6);
					c3Layer[i].setNumber(j, k, MathFunction.ReLU(temp / 6));
				}
			}
		}
		//Calculate NO.4 layer: s4 layer
		for(int i = 0;i < 16;i++){
			for(int j = 0;j < 5;j++){
				for(int k = 0;k < 5;k++){
					int j2 = j * sk4[i].getStride();
					int k2 = k * sk4[i].getStride();
					temp = MathFunction.calculateSubsamplePoint(c3Layer[i],j2, k2, sk4[i]);
					s4Layer[i].setRawValue(j, k, temp);
					s4Layer[i].setNumber(j, k, MathFunction.ReLU(temp));
				}
			}
		}
		//Calcualte NO.5 layer: c5 layer  
		gapLine = MathFunction.mapToLineNumber(s4Layer);
		gapRawNumber = MathFunction.mapToLineRawNumber(s4Layer);
		for(int i = 0;i < 120;i++){
			double tempHidden = 0;
			for(int j = 0;j < 16 * 5 * 5;j++){
				tempHidden += gapLine[j] * gap5[i].getWeight(j);
			}
			tempHidden = tempHidden + gap5[i].getBias();
			c5Layer[i].setRawValue(0, 0, tempHidden);
			c5Layer[i].setNumber(0, 0, MathFunction.ReLU(tempHidden));
		}
		//Calculate NO.6 layer: f6 layer
		for(int i = 0;i < 84;i++){
			double tempHidden = 0;
			for(int j = 0;j < 120;j++){
				tempHidden += bp1[i].getWeight(j) * c5Layer[j].getNumber(0, 0);
			}
			tempHidden = tempHidden + bp1[i].getBias();
			f6Layer[i].setRawValue(0, 0, tempHidden);
			f6Layer[i].setNumber(0, 0, MathFunction.ReLU(tempHidden)); 
		}
		//Calculate NO.7 layer: output layer   
		for(int i = 0;i < 10;i++){
			double tempOutput = 0;
			for(int j = 0;j < 84;j++){
				tempOutput += bp2[i].getWeight(j) * f6Layer[j].getNumber(0, 0);
			}
			tempOutput = tempOutput + bp2[i].getBias();
			outputLayer[i].setRawValue(0, 0, tempOutput);
			outputLayer[i].setNumber(0, 0, MathFunction.sigmoid(tempOutput)); 
			System.out.println(MathFunction.sigmoid(tempOutput));
		}
		System.out.println();
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
		calculateSensitivation();
	}
	
	private void calculateSensitivation(){
		//首先计算最后一层output层的Δerror
		for(int i = 0;i < 10;i++){
			double loss = MathFunction.sigmoidDerivation(outputLayer[i].getRawValue(0, 0)) 
					* (desiredOutput[i] - outputLayer[i].getNumber(0, 0));
			outputLayer[i].setError(0, 0, loss);
			
			//开始调整最下层weight
			for(int j = 0;j < 84;j++){
				bp2[i].setWeight(j, (0 - learningRate) * f6Layer[j].getNumber(0, 0) * loss + bp2[i].getWeight(j));
			}
			bp2[i].setBias((0 - learningRate) * loss + bp2[i].getBias());
			
		}
		//然后计算倒数第二层f6的Δerror
		for(int i = 0;i < 84;i++){
			double loss = 0;
			for(int j = 0;j < 10;j++){
				loss += bp2[j].getWeight(i) * outputLayer[j].getError(0, 0)
						* MathFunction.ReLUDerivation(f6Layer[i].getRawValue(0, 0));
			}
			f6Layer[i].setError(0, 0, loss);
			
			//开始调整下层weight
			for(int j = 0;j < 120;j++){
				bp1[i].setWeight(j,(0 - learningRate) * c5Layer[j].getNumber(0, 0) * loss + bp1[i].getWeight(j));
			}
			bp1[i].setBias((0 - learningRate) * loss + bp1[i].getBias());
			
		}
		//然后计算倒数第三层C5的Δerror
		for(int i = 0;i < 120;i++){
			double loss = 0;
			for(int j = 0;j < 84;j++){
				loss += bp1[j].getWeight(i) * f6Layer[j].getError(0, 0)
						* MathFunction.ReLUDerivation(c5Layer[i].getRawValue(0, 0));
			}
			c5Layer[i].setError(0, 0, loss);
			
			for(int j = 0;j < 400;j++){
				gap5[i].setWeight(j, (0 - learningRate) * gapLine[j] * loss + gap5[i].getWeight(j));
			}
			gap5[i].setBias((0 - learningRate) * loss + gap5[i].getBias());
			
		}
		//然后先计算线性矩阵的Δerror，然后将现行序列展开到16个5 * 5的矩阵组当中,也就是S4的Δerror
		double[] gapLoss = new double[16 * 5 *5];
		for(int i = 0;i < 16 * 5 * 5;i++){
			double loss = 0;
			for(int j = 0;j < 120;j++){
				loss += gap5[j].getWeight(i) * c5Layer[j].getError(0, 0)
						* MathFunction.ReLUDerivation(gapRawNumber[i]);
			}
			gapLoss[i] = loss;
		}
		for(int i = 0;i < 16;i++){
			double tempBiasChange = 0;
			double tempWeightChange = 0;
			for(int j = 0;j < 5;j++){
				for(int k = 0;k < 5;k++){
					s4Layer[i].setError(j, k, gapLoss[i * 16 + j * 5 + k]);
					tempBiasChange += s4Layer[i].getError(j, k);
					tempWeightChange += s4Layer[i].getError(j, k) * s4Layer[i].getNumber(j, k);
				}
			}
			sk4[i].setBeta((0 - learningRate) * tempWeightChange + sk4[i].getBeta());
			sk4[i].settBias((0 - learningRate) * tempBiasChange + sk4[i].getBias());
		}
		
		//然后计算C3的Δerror
		for(int i = 0;i < 16;i++){
			double[][] newMatrix = MathFunction.spandBackpropagationWeight(s4Layer[i].getMatrix(), 2);
			for(int j = 0;j < 10;j++){
				for(int k = 0;k < 10;k++){
					c3Layer[i].setError(j, k, sk4[i].getBeta() 
							* newMatrix[j][k] * MathFunction.ReLUDerivation(c3Layer[i].getRawValue(j, k)));
				}
			}
		}
		
		for(int indexCK = 0;indexCK < 16;indexCK++){
			for(int i = 0;i < 5;i++){
				for(int j = 0;j < 5;j++){
					double tempWeightChange = 0;
					double tempBiasChange = 0;
					for(int m = 0;m < 10;m++){
						for(int n = 0;n < 10;n++){
							tempBiasChange += c3Layer[indexCK].getError(m, n);
							for(int indexMap = 0;indexMap < 6;indexMap++){
								tempWeightChange += s2Layer[indexMap].getNumber(m + i, n + j) * c3Layer[indexCK].getError(m, n);
							}
							
						}
					}
					ck3[indexCK].setBias( (0 - learningRate) * tempBiasChange + ck3[indexCK].getBias());
					ck3[indexCK].setWeight(i, j, (0 - learningRate) * tempWeightChange + ck3[indexCK].getWeight(i, j));
				}
			}
		}
		
		
		
		
		//然后计算s2的Δerror
		for(int i = 0;i < 16;i++){
			for(int j = 0;j < 10;j++){
				for(int k = 0;k < 10;k++){
					for(int m = 0; m < 6;m++){
						MathFunction.calculateConvolutionalReverse(c3Layer[i], s2Layer[m], j, k, ck3[m]);
					}	
				}
			}
		}
		for(int i = 0;i < 6;i++){
			double tempBiasChange = 0;
			double tempWeightChange = 0;
			for(int j = 0;j < 14;j++){
				for(int k = 0;k < 14;k++){
					tempBiasChange += s2Layer[i].getError(j, k);
					tempWeightChange += s2Layer[i].getError(j, k) * s2Layer[i].getNumber(j, k);
				}
			}
			sk2[i].setBeta((0 - learningRate) * tempWeightChange + sk2[i].getBeta());
			sk2[i].settBias(- learningRate * tempBiasChange + sk2[i].getBias());
		}
		//然后计算C1的Δerror
		//to-do
		for(int i = 0;i < 6;i++){
			double[][] newMatrix = MathFunction.spandBackpropagationWeight(s2Layer[i].getMatrix(), 2);
			double tempBiasChange = 0;
			for(int j = 0;j < 28;j++){
				for(int k = 0;k < 28;k++){
					c1Layer[i].setError(j, k, sk2[i].getBeta() 
							* newMatrix[j][k] * MathFunction.ReLUDerivation(c1Layer[i].getRawValue(j, k)));
					tempBiasChange += c1Layer[i].getError(j, k);
				}
			}
			ck1[i].setBias((0 - learningRate) * tempBiasChange + ck1[i].getBias());
		}
		
		for(int indexCK = 0;indexCK < 6;indexCK++){
			for(int i = 0;i < 5;i++){
				for(int j = 0;j < 5;j++){
					double tempWeightChange = 0;
					for(int m = 0;m < 28;m++){
						for(int n = 0;n < 28;n++){
							tempWeightChange += inputLayer[0].getNumber(m + i, n + j) * c1Layer[indexCK].getError(m, n);
						}
					}
					ck1[indexCK].setWeight(i, j, (0 - learningRate) * tempWeightChange + ck1[indexCK].getWeight(i, j));
				}
			}
		}
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
