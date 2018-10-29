package neuralnet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Set;

public class Nnet {
	private int numinput;
	private int numoutput;
	private int numhiddenlayer;
	private double[][] inputvalues;
	private double[] trueY;
	private int numneuron;
	private ArrayList<Layer> layerlist;

	private double cost;
	private double[][] ychanged;
	private double lambda;
	private int maxit;
	private ArrayList<Double> costlist;
	private double lr;

	public Nnet( int Numhiddenlayer, double[][] InputValues, double[] TrueY, int NumN,double Lambda, int Maxit, double LR) {
		this.numhiddenlayer = Numhiddenlayer;
		this.inputvalues = InputValues;
		this.trueY = TrueY;
		this.numneuron = NumN;
		this.numoutput = this.GetNumOutput() ;
		this.layerlist = new ArrayList<Layer>();

		this.cost = 0;
		this.ychanged = Changeychanged();
		this.lambda = Lambda;
		this.maxit = Maxit;
		this.costlist  = new ArrayList<Double>();
		this.lr = LR;
	}
	
	public double[][] getinpv(){
		return this.inputvalues;
	}
	
	public double[][] Changeychanged(){
		double[][] yc = new double[this.trueY.length][this.numoutput];
		for(int i =0; i< this.trueY.length; i++) {
			int numb = (int) this.trueY[i];
			yc[i][numb] = (double) 1;
		}
		return yc;
	}

	public double[][] propogateforward(){
		
		if (this.layerlist.size() == 0){
			Layer initi = new InputLayer(this.inputvalues, this.numneuron, this.lambda, this.lr);
			initi.Propagateforward();
			this.layerlist.add(initi);
			HiddenLayer middle = new HiddenLayer(this.numneuron,initi,1,this.numneuron,this.lambda,this.lr);
			middle.Propagateforward();
			this.layerlist.add(middle);
			for(int i = 2; i - 1 < this.numhiddenlayer; i++) {
				middle = new HiddenLayer(this.numneuron,middle,i,this.numneuron,this.lambda,this.lr);
				middle.Propagateforward();
				this.layerlist.add(middle);
			}
			OutputLayer fin = new OutputLayer(middle,this.numoutput, this.ychanged,this.lambda,this.lr);
			fin.Propagateforward();
			this.layerlist.add(fin);
			this.Costfunction(fin.Getoutput());
			System.out.println(this.cost+" this is cost");
			this.costlist.add(this.cost);
			return fin.Getoutput();
			
		}
		Layer initi = this.layerlist.get(0);
		initi.Propagateforward();
		for(int i =1; i < this.layerlist.size()-1; i ++) {
			Layer middle = layerlist.get(i);
			middle.Propagateforward();
		}
		Layer fin  = this.layerlist.get(this.layerlist.size()-1);
		fin.Propagateforward();	
		this.Costfunction(fin.Getoutput());
		System.out.println(this.cost+" this is cost");
		this.costlist.add(this.cost);
		return fin.Getoutput();
	}
	
	public void Propagateback() {
		//System.out.println("final start");
		int i = this.layerlist.size() -1;
		Layer fin = this.layerlist.get(i);
		fin.Propagateback();
		double[][] diff = fin.Getdiff();
		i -= 1;
		//System.out.println("final done");
		while(i > 0) {
			Layer hidden = this.layerlist.get(i);
			hidden.Setdiff(diff);
			hidden.Propagateback();
			diff = hidden.Getdiff();
			i -= 1;
		}
		//System.out.println("middle done");
		Layer input = this.layerlist.get(0);
		input.Setdiff(diff);
		input.Propagateback();
		//System.out.println("one epoch done");
	}
	
	public int GetNumOutput() {
		 Set<Double> linkedHashSet = new LinkedHashSet<>();
		 for (double d: this.trueY) {
			 linkedHashSet.add(d);
		 }
		 return linkedHashSet.size();

	}
	
	public double Getregulation() {
		ArrayList<double[][]> weights = new ArrayList<double[][]>();
		double regvalue = 0.0;
		for(Layer la: this.layerlist) {
			weights.add(la.getweight());
		}
		for(double[][] weight: weights) {
			regvalue += CalcRegW(weight);
		}
		regvalue *= this.lambda / (2 * this.inputvalues.length) ;
		return regvalue;
	}
	
	public double CalcRegW(double[][] weight) {
		double result = 0.0;
		for(double[] row: weight) {
			for(int i = 1; i< row.length-1; i++) {
				result += row[i] * row[i];
			}
		}
		return result;
	}
	
	
	public void Costfunction(double[][] output) {
		for(int i=0; i< output.length; i++ ) {
			for( int j = 0; j< output[0].length; j++) {
				this.cost += Math.log(output[i][j]) * (-1) * this.ychanged[i][j] -Math.log(1-output[i][j]) * (1 - this.ychanged[i][j]);
			}
		}
		this.cost = this.cost/this.inputvalues.length + this.Getregulation();
		
	}
	
	public double GetCost() {
		return this.cost;
	}
	
	public double[][] Start() {
		int i  = 0;
		while(i < this.maxit-1) {
			//System.out.println(i+ " this is i hhhh");
			this.propogateforward();
			this.Propagateback();
			i += 1;
		}
		this.propogateforward();
		return this.propogateforward();
	}
	
	public static void main(String[] args) {
		double[][] X = new double[7][2];
		X[0][0] = 1.0;
		X[0][1] = 1.0;
		X[1][0] = 1.0;
		X[1][1] = 0.0;
		X[2][0] = 0.0;
		X[2][1] = 1.0;
		X[3][0] = 0.0;
		X[3][1] = 0.0; 
		X[4][0] = 0.0;
		X[4][1] = 0.0; 
		X[5][0] = 0.0;
		X[5][1] = 0.0; 
		X[6][0] = 0.0;
		X[6][1] = 0.0; 
		
		double[] Y = new double[7];
		Y[0] = 1;

		Y[1] = 0;

		Y[2] = 0;
		Y[3] = 1;
		Y[4] = 1;
		Y[5] = 1;
		Y[6] = 1;

		Nnet nn = new Nnet(4, X, Y,3,1,50, 0.05);
		double[][] k =  nn.Start();
		
		
		for (double[] d : k) {
			for(double dd: d) {
				System.out.print(dd);
			}
			System.out.println(" ");
		}

 	}
	
}
