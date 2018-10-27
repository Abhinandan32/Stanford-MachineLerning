package neuralnet;

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
	
	// input layer
	public Layer(double[][] Inputvalue, int Numoutput, double Lambda) {
		this.inputvalue = Inputvalue;
		this.numoutput = Numoutput;
		this.layerindex = 0;
		this.outputvalue = new double[Inputvalue.length][this.numoutput];
		this.lambda = Lambda;
	}
	
	// hidden layer
	public Layer( int Numoutput, Layer prevLayer, int Index, int Numneuron, double Lambda) {
		this.numoutput = Numoutput;
		this.prevlayer = prevLayer;
		this.numneuron = Numneuron;
		this.layerindex = Index;
		this.lambda = Lambda;
	}
	
	// output layer
	public Layer( Layer prevLayer, int numoutput, double[][] Truey, double Lambda) {
		this.numoutput = numoutput;
		this.prevlayer = prevLayer;
		this.trueY = Truey;
		this.lambda = Lambda;
	}
	
	public double[][] GetTrueY(){
		return this.trueY;
	}
	
	public void Setdiff(double[][] dif) {
		this.diff = dif;
	}
	public double[][] Getdiff(){
		return this.diff;
	}
	
	public void Initweight() {
	}
	
	public void Propagateforward() {
	}
	
	public void Propagateback() {
	}
	
	public double[][] Getinput(){
		return this.inputvalue;
	}
	
	public void sigmoid() {
		for (int i = 0 ; i < this.outputvalue.length; i ++ ) {
			for( int j = 0; j < outputvalue[i].length; j ++) {
				this.outputvalue[i][j] = 1/(1 + Math.exp(- this.outputvalue[i][j]));
			}
		}
	}
	public double[][] Getprevoutput() {
		return this.prevlayer.outputvalue;
	}
	
	public double[][] getweight(){
		return this.thisweight;
	}
	
	public int getlayerindex() {
		return this.layerindex;
	}
	
	
	public int Getnoutput() {
		return this.numoutput;
	}
	public int GetNneuron() {
		return this.numneuron;
	}
	
	public void Setweightinit(int m, int n) {
		Random r = new Random();
		this.thisweight = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				this.thisweight[i][j] = r.nextDouble();
			}
		}
	}
	
	public void Updateweight(double[][] weight) {
		this.thisweight = weight;
	}
	
	public void Setoutputv(double[][] s) {
		this.outputvalue = s;
	}
	
	public double[][] Getoutput(){
		return this.outputvalue;
	}
	

	
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
					//System.out.println(inputvalue[i][k]* thisweight[j][k]+"result");
					}
				outputvalue[i][j] = result;
			}
		}
		this.Setoutputv(outputvalue);
		this.sigmoid();
		
	}
	
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
	
	public void castfirst() {
		double[][] tempdiff = new double[this.diff.length][this.diff[0].length-1];
		for(int i =0 ; i < tempdiff.length; i++) {
			for(int j = 0; j < tempdiff[0].length ; j++ ) {
				tempdiff[i][j] = this.diff[i][j+1];
			}
		}
		this.Setdiff(tempdiff);
	}
	
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
	
	public void calcgrad(double[][] diff, double[][] input) {
		int m = this.getweight().length;
		int n = this.getweight()[0].length;
		this.weightgrad = new double[m][n];
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
				this.weightgrad[i][j] = result/input.length;
			}
		}
		double[][] reg = this.calcgradreg(input);
		for (int i = 0; i< reg.length ; i++) {
			for(int j = 0; j <reg[0].length; j++) {
				this.weightgrad[i][j] += reg[i][j];
			}
		}
		this.Updateweight(this.weightgrad);
	}
	
	public double[][] calcgradreg(double[][] input){
		double[][] reg = new double[this.thisweight.length][this.thisweight[0].length];
 		for(int i = 0; i< this.thisweight.length; i++) {
 			reg[i][0] = 0;
			for(int j = 1; j < this.thisweight[0].length; j++) {
				reg[i][j] = this.lambda / input.length * reg[i][j] * reg[i][j];
			}
		}
 		return reg;
	}
	


}
