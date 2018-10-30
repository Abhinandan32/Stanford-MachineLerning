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
	private ArrayList<double[][]> finalweight;
	private double[][] finaloutput;

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
		this.finalweight = new ArrayList<double[][]>();
	}
	
	public double[][] getinpv(){
		return this.inputvalues;
	}
	
	// Change Y from an m x 1 matrix to an m x b matrix where b is the number of distinct values of Y from 0 to b-1 
	public double[][] Changeychanged(){
		double[][] yc = new double[this.trueY.length][this.numoutput];
		for(int i =0; i< this.trueY.length; i++) {
			int numb = (int) this.trueY[i];
			yc[i][numb] = (double) 1;
		}
		return yc;
	}
	
	// propogateforward, from input layer to outputlayer
	public double[][] propogateforward(){
		
		if (this.layerlist.size() == 0){
			Layer initi = new InputLayer(this.inputvalues, this.numneuron, this.lambda, this.lr);
			initi.Propagateforward();
			this.finalweight.add(initi.getweight());
			//System.out.println(initi.getlayerindex());

			this.layerlist.add(initi);
			HiddenLayer middle = new HiddenLayer(this.numneuron,initi,1,this.numneuron,this.lambda,this.lr);
			middle.Propagateforward();
			//System.out.println(middle.getlayerindex());
			this.finalweight.add(middle.getweight());
			this.layerlist.add(middle);
			for(int i = 2; i < this.numhiddenlayer; i++) {
				middle = new HiddenLayer(this.numneuron,middle,i,this.numneuron,this.lambda,this.lr);
				middle.Propagateforward();
				this.finalweight.add(middle.getweight());
				this.layerlist.add(middle);
				//System.out.println(middle.getlayerindex());
			}
			LastHiddenLayer fin = new LastHiddenLayer(middle,this.numoutput, this.ychanged,this.lambda,this.lr);
			fin.Propagateforward();
			int Index = this.numhiddenlayer;
			fin.setlayeriindex(Index);
			//System.out.println(fin.getlayerindex());
			this.finalweight.add(fin.getweight());
			this.layerlist.add(fin);
			this.Costfunction(fin.Getoutput());
			//System.out.println(this.cost+" this is cost");
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
		//System.out.println(this.cost+" this is cost");
		this.costlist.add(this.cost);
		return fin.Getoutput();
	}
	
	public void Setinput(double[][] input) {
		this.inputvalues = input;
	}
	
	public void Setoutput(double[] output) {
		this.trueY  = output;
		this.ychanged = this.Changeychanged();
	}
	
	public double testset(double[][] input, double[] outcome) {
		this.Setinput(input);
		this.Setoutput(outcome);
		Layer initi = new InputLayer(this.inputvalues, this.numneuron, this.lambda, this.lr);
		initi.Updateweight(this.finalweight.get(0));
		initi.Propagateforward();
		//System.out.println(initi.getlayerindex());
		HiddenLayer middle = new HiddenLayer(this.numneuron,initi,1,this.numneuron,this.lambda,this.lr);
		middle.Updateweight(this.finalweight.get(1));
		middle.Propagateforward();
		//System.out.println(middle.getlayerindex());
		for(int i = 2; i < this.numhiddenlayer; i++) {
			middle = new HiddenLayer(this.numneuron,middle,i,this.numneuron,this.lambda,this.lr);
			middle.Updateweight(this.finalweight.get(i));
			middle.Propagateforward();
			//System.out.println(middle.getlayerindex());
		}
		LastHiddenLayer fin = new LastHiddenLayer(middle,this.numoutput, this.ychanged,this.lambda,this.lr);
		fin.Updateweight(this.finalweight.get(this.numhiddenlayer));
		fin.Propagateforward();
		this.cost = 0;
		this.Costfunction(fin.Getoutput());
		//System.out.println(this.cost+" this is cost");
		return this.cost;
	}
	
	
	
	public void Propagateback() {
		//System.out.println("final start");		
		int i = this.layerlist.size() -1;
		Layer fin = this.layerlist.get(i);
		fin.Propagateback();
		this.finalweight.set(i, fin.getweight());
		double[][] diff = fin.Getdiff();
		i -= 1;
		//System.out.println("final done");
		while(i > 0) {
			Layer hidden = this.layerlist.get(i);
			hidden.Setdiff(diff);
			hidden.Propagateback();
			this.finalweight.set(i, hidden.getweight());
			diff = hidden.Getdiff();
			i -= 1;
		}
		//System.out.println("middle done");
		Layer input = this.layerlist.get(0);
		input.Setdiff(diff);
		input.Propagateback();
		this.finalweight.set(0, input.getweight());
		//System.out.println("one epoch done");
	}
	
	public int GetNumOutput() {
		 Set<Double> linkedHashSet = new LinkedHashSet<>();
		 for (double d: this.trueY) {
			 linkedHashSet.add(d);
		 }
		 return linkedHashSet.size();

	}
	
	public ArrayList<double[][]> Getfinalweight(){
		return this.finalweight;
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
	
	public double Getcost() {
		return this.cost;
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
	
	public void Start() {
		int i  = 0;
		while(i < this.maxit) {
			//System.out.println(i+ " this is i hhhh");
			this.finaloutput = this.propogateforward();
			this.Propagateback();
			i += 1;
		}
		this.propogateforward();
	}
	
	public double[][] GetfinalOutput(){
		return this.finaloutput;
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

		Nnet nn = new Nnet(4, X, Y,3,1,500, 0.05);
		nn.Start();
		double[][] k = nn.GetfinalOutput();
		
		
		for (double[] d : k) {
			for(double dd: d) {
				System.out.print(dd);
			}
			System.out.println(" ");
		}

 	}
	
}
