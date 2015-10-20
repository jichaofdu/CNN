package CNN;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

public class IHFormulaParameter implements Serializable{

	private static final long serialVersionUID = -8907566529782739717L;
	private int length;
	private double[] weight;
	private double bias;
	private int index;
	private double[] change;

	public IHFormulaParameter(int length,double[] paraSet,double bias,int index){
		this.length = length;
		this.bias = bias;
		this.index = index;
		this.weight = new double[length];
		this.change = new double[length];
		if(paraSet != null){
			for(int i = 0;i < length;i++){
				this.weight[i] = paraSet[i];
				this.change[i] = 0;
			}
		}
	}
	
	public void writeToDiskIH(String path) throws FileNotFoundException, IOException{
		String fileName = path + "IH" + index + ".obj";
		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(fileName));
		out.writeObject(this);
		out.close();
	}
	
	public void readFromDiskIH(String path) throws FileNotFoundException, IOException, ClassNotFoundException{
		String fileName = path + "IH" + index + ".obj";
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName));
		IHFormulaParameter newRead = (IHFormulaParameter)in.readObject();
		this.bias = newRead.getBias();
		this.length = newRead.getLength();
		this.index = newRead.getIndex();
		for(int i = 0;i < length;i++){
			this.weight[i] = newRead.getWeight(i);
			this.change[i] = newRead.getChange(i);
		}
		in.close();
	}
	
	public double getChange(int i){
		return this.change[i];
	}
	
	public int getLength(){
		return this.length;
	}
	
	public int getIndex(){
		return index;
	}
	
	public double getWeight(int paraIndex){
		return weight[paraIndex];
	}
	
	public double getBias(){
		return bias;
	}
	
	public void setBias(double bias){
		this.bias = bias;
	}
	
	public void setWeight(int i,double weight){
		this.weight[i] = weight;
	}
	
	public void setChange(int i,double change){
		this.change[i] = change;
	}
	
}
