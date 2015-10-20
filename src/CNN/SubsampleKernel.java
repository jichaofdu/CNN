package CNN;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

public class SubsampleKernel  implements Serializable{

	private static final long serialVersionUID = -3382071696877307147L;
	private double beta;
	private double bias;
	private String tag;
	private double change;
	private int stride;
	
	public SubsampleKernel(double beta,double bias,int stepLength,String tag){
		this.beta = beta;
		this.bias = bias;
		this.tag = tag;
		this.change = 0;
		this.stride = stepLength;
	}
	
	public int getStride() {
		return stride;
	}

	public double getBeta(){
		return beta;
	}
	
	public double getBias(){
		return bias;
	}
	
	public void setBeta(double newBeta){
		this.beta = newBeta;
	}
	
	public void settBias(double newBias){
		this.bias = newBias;
	}
	
	public String getTag(){
		return this.tag;
	}
	
	public double getChange(){
		return change;
	}
	
	public void setChange(double newChange){
		this.change = newChange;
	}
	
	public void writeToDiskSK(String path) throws FileNotFoundException, IOException{
		String fileName = this.tag;
		fileName = path + fileName + ".obj";
		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(fileName));
		out.writeObject(this);
		out.close();
	}
	
	public void readFromDiskHO(String path) throws FileNotFoundException, IOException, ClassNotFoundException{
		String fileName = this.tag;
		fileName = path + fileName + ".obj";
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName));
		SubsampleKernel newRead = (SubsampleKernel)in.readObject();
		this.beta = newRead.getBeta();
		this.bias = newRead.getBias();
		this.tag = newRead.getTag();
		this.change = newRead.getChange();
		in.close();
	}
}
