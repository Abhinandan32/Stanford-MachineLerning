package neuralnet;

import java.util.ArrayList;
import java.util.Random;

public class Layer {
	private double[][] inputvalue;
	private int numoutput;
	private Layer prevlayer;
	private Layer nextlayer;
	private int layerindex;
	private int numneuron;
	private double[][] outputvalue;
	private double[][] thisweight;
	private double[][] diff;
	private double[][] trueY;
	private double lambda;
	private double[][] weightgrad;
	private double lr; 
	
	
	// input layer
	public Layer(double[][] Inputvalue, int Numoutput, double Lambda,double LR) {
		this.inputvalue = Inputvalue;
		this.numoutput = Numoutput;
		this.layerindex = 0;
		this.outputvalue = new double[Inputvalue.length][this.numoutput];
		this.lambda = Lambda;
		this.lr = LR;

	}
	
	// hidden layer
	public Layer( int Numoutput, Layer prevLayer, int Index, int Numneuron, double Lambda,double LR) {
		this.numoutput = Numoutput;
		this.prevlayer = prevLayer;
		this.numneuron = Numneuron;
		this.layerindex = Index;
		this.lambda = Lambda;
		this.lr = LR;
	}
	
	// Last hidden layer, the output of this layer is the output 
	public Layer( Layer prevLayer, int numoutput, double[][] Truey, double Lambda,double LR) {
		this.numoutput = numoutput;
		this.prevlayer = prevLayer;
		this.trueY = Truey;
		this.lambda = Lambda;
		this.lr = LR;
	}
	
	// get Y in m x b form
	public double[][] GetTrueY(){
		return this.trueY;
	}
	
	// set delta
	public void Setdiff(double[][] dif) {
		this.diff = dif;
	}
	
	// get delta
	public double[][] Getdiff(){
		return this.diff;
	}
	
	// initiation of weight/theta
	public void Initweight() {
	}
	
	
	public void Propagateforward() {
	}
	
	public void Propagateback() {
	}
	
	//get the input of the layer
	public double[][] Getinput(){
		return this.inputvalue;
	}
	
	//sigmoid/logistic function
	public void sigmoid() {
		for (int i = 0 ; i < this.outputvalue.length; i ++ ) {
			for( int j = 0; j < outputvalue[i].length; j ++) {
				this.outputvalue[i][j] = 1/(1 + Math.exp(- this.outputvalue[i][j]));
			}
		}
	} 
	
	//get the output of the previous layer
	public double[][] Getprevoutput() {
		return this.prevlayer.outputvalue;
	}
	
	//get the weight of current layer
	public double[][] getweight(){
		return this.thisweight;
	}
	
	//get the index of current layer
	public int getlayerindex() {
		return this.layerindex;
	}
	
	public void setlayeriindex(int Index) {
		this.layerindex = Index;
	}
	// get the number of output we would have in current layer.
	//in hidden layer, it is the number of neurons, in the last hidden layer it is the number of 
	//distinct value of Y.
	public int Getnoutput() {
		return this.numoutput;
	}
	
	// get the number of neuron in current layer
	public int GetNneuron() {
		return this.numneuron;
	}
	
	public void SetWeight(double[][] w) {
		this.thisweight = w;
	}
	
	//initiation of weight, called by Initweight
	public void Setweightinit(int m, int n) {
		Random r = new Random();
		this.thisweight = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				this.thisweight[i][j] = 2 * r.nextDouble()-1;
			}
		}
	}
	
	
	//Update the weight with new value
	public void Updateweight(double[][] weight) {
		this.thisweight = weight;
	}
	
	// set the output value 
	public void Setoutputv(double[][] s) {
		this.outputvalue = s;
	}
	
	// get the output value
	public double[][] Getoutput(){
		return this.outputvalue;
	}
	

	// apply forward propogation by Theta * inputvalue and apply sigmoid function 
	public void calculateProp(double[][] inputvalue, double[][] thisweight, int noutput) {
		int numoutput = this.Getnoutput();
		double[][] outputvalue = new double[inputvalue.length][numoutput];
		for(int i = 0; i < inputvalue.length ; i++) {
			for (int j = 0; j < noutput; j ++) {
				double result = 0;
				for(int k = 0; k< thisweight[0].length; k++) {
					if (k == 0) {
						result += 1 * thisweight[j][k];
					}
					else {
						result += inputvalue[i][k-1]* thisweight[j][k];
					}
				}
				outputvalue[i][j] = result;
			}
		}
		this.Setoutputv(outputvalue);
		this.sigmoid();

		
	}
	
	//Used for back propogation
	public double[][] SigmoidGradient() {
		int m = this.Getprevoutput().length;
		int n = this.Getprevoutput()[0].length;
		double[][] z = this.Getprevoutput();
		double[][]SG = new double[m][1+n];
		//System.out.println("calcgrad done!????!");
		for(int i = 0; i < m ; i++) {
			//System.out.println("calcgrad done!!"+ i);
			SG[i][0] = 1;
			for( int j = 1 ; j < 1+n; j++) {
				//System.out.println("calcgrad done!!??????"+ j);
				SG[i][j] = z[i][j-1] * (1- z[i][j-1]);
			}
		}
		return SG;
	}
	
	public void printdata(double[][] yc ) {
		for(double[] y: yc) {
			for(double yy : y) {
				System.out.print(yy+ " ");
			}
			System.out.println(" " );
		}
	}
	// calculated the delta of previous hidden layer, used in back propogation.
	public void calchiddendiff(){
		this.diff = this.multiply(this.diff, this.thisweight);
		//System.out.println("calcgrad done!!");
		double[][] SG = this.SigmoidGradient();
		//System.out.println("calcgrad done!!?");
		for( int i  = 0; i < this.diff.length; i++) {
			for (int j = 0; j < SG[0].length; j++) {
				this.diff[i][j] *= SG[i][j];
			}
		}
		//System.out.println("calcgrad done!!!");
		this.castfirst();
		
	}
	
	// remove the first column which is the column for bias.
	public void castfirst() {
		double[][] tempdiff = new double[this.diff.length][this.diff[0].length-1];
		for(int i =0 ; i < tempdiff.length; i++) {
			for(int j = 0; j < tempdiff[0].length ; j++ ) {
				tempdiff[i][j] = this.diff[i][j+1];
			}
		}
		this.Setdiff(tempdiff);
	}
	
	// apply multiplication between two matrices
	public double[][] multiply(double[][] df, double[][] wei){
		int m = df.length;
		int n = wei[0].length;
		double[][] dff = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				double result= 0;
				for (int k = 0; k < wei.length ; k++) {
					result += df[i][k] * wei[k][j];
				}
				dff[i][j] += result;
			}
		}
		return dff;
	}
	public void Initweightgrad() {
		this.weightgrad = new double[this.getweight().length][this.getweight()[0].length];
	}
	
	// calculate the gradient descent of current layer and update weight, regulation is included.
	public void calcgrad(double[][] diff, double[][] input) {
		int m = this.getweight().length;
		int n = this.getweight()[0].length;
		for( int i = 0; i < m; i++) {
			//System.out.println("this is i"+ i);
			for (int j = 0; j < n; j++) {
				//System.out.println("this is j" + j);
				double result = 0;
				for(int k =0; k< diff.length; k++) {
					//System.out.println("this is k" + k);
					if (j ==0) {
						//System.out.println("should get one"+ diff[k][i] );
						result += diff[k][i] * 1;
					}else {
						//System.out.println("should get one"+ diff[k][i] );
						result += diff[k][i] * input[k][j-1];
					}
				}
				this.weightgrad[i][j] += result;
				this.weightgrad[i][j] /= input.length;
			}
		}
		
		double[][] reg = this.calcgradreg(input);
		for (int i = 0; i< reg.length ; i++) {
			for(int j = 1; j <reg[0].length; j++) {
				this.weightgrad[i][j] += reg[i][j-1];
			}
		}
		this.Updateweight(this.backweight(this.weightgrad));
	}
	
	// apply gradient descent and learning rate to update weight
	public double[][] backweight(double[][] weightgrad) {
		double[][] tempweight = this.thisweight;
		for(int i = 0; i < weightgrad.length; i++) {
			for(int j = 0; j < weightgrad[0].length; j++) {
				tempweight[i][j] = tempweight[i][j] - this.lr * weightgrad[i][j];
			}
		}
		//System.out.println("Start changing weight!!");
		//this.printdata(this.thisweight);
		//System.out.println("Done!!!");
		return tempweight;
	}
	
	// calculate the regulation for gradient descent 
	public double[][] calcgradreg(double[][] input){
		double[][] reg = new double[this.thisweight.length][this.thisweight[0].length];
 		for(int i = 0; i< this.thisweight.length; i++) {
 			reg[i][0] = 0;
			for(int j = 1; j < this.thisweight[0].length; j++) {
				reg[i][j] = this.lambda / input.length * reg[i][j];
			}
		}
 		return reg;
	}
	


}
