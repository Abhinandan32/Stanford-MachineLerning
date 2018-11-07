package neuralnet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
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
	private double lr;
	private ArrayList<double[][]> finalweight;
	private double[][] finaloutput;
	private ArrayList<Double> predcosts;
	private double predcost;
	

	

	public Nnet( int Numhiddenlayer, double[][] InputValues, double[] TrueY, int NumN,double Lambda, int Maxit, double LR) {
		this.numhiddenlayer = Numhiddenlayer;
		this.inputvalues = InputValues;
		this.trueY = TrueY;
		this.numneuron = NumN;
		this.numoutput = this.GetNumOutput() ;
		this.layerlist = new ArrayList<Layer>();

		this.cost = 0;
		this.ychanged = Changeychanged(this.trueY);
		this.lambda = Lambda;
		this.maxit = Maxit;
		this.lr = LR;
		this.finalweight = new ArrayList<double[][]>();
		this.predcosts = new ArrayList<Double>(); 

		
	}
	
	public double[][] getinpv(){
		return this.inputvalues;
	}
	
	// Change Y from an m x 1 matrix to an m x b matrix where b is the number of distinct values of Y from 0 to b-1 
	public double[][] Changeychanged(double[] trueY){
		double[][] yc = new double[trueY.length][this.numoutput];
		for(int i =0; i< trueY.length; i++) {
			int numb = (int) trueY[i];
			yc[i][numb] = (double) 1;
		}
		//System.out.println(" ChangedYc");
		//this.printdata(yc);
		return yc;
	}
	
	
	public void printdata(double[][] yc ) {
		for(double[] y: yc) {
			for(double yy : y) {
				System.out.print(yy+ " ");
			}
			System.out.println(" " );
		}
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
			this.cost =  this.Costfunction(fin.Getoutput(),this.ychanged,this.inputvalues.length);
			//System.out.println(this.cost+" this is cost");
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
		this.cost = this.Costfunction(fin.Getoutput(), this.ychanged,this.inputvalues.length);
		//System.out.println(this.cost+" this is cost");
		return fin.Getoutput();
	}
	
	public void Setinput(double[][] input) {
		this.inputvalues = input;
	}
	
	public void Setoutput(double[] output) {
		this.trueY  = output;
		this.ychanged = this.Changeychanged(this.trueY);
	}
	
	public double[][] predict(double[][] input, double[] y) {
		this.Setinput(input);
		double[][] changedy = this.Changeychanged(y);
		Layer l = this.layerlist.get(0);
		for(int i = 1; i < this.layerlist.size()-1; i++) {
			l = this.layerlist.get(i);
			l.SetWeight(this.finalweight.get(i));
			l.Propagateforward();
		}
		double[][] out = l.Getoutput();
		double[][] outp = new double[changedy.length][out[0].length-1];
		for(int i = 0 ; i < changedy.length; i++) {
			for( int j = 1; j < out[0].length; j ++) {
				outp[i][j-1] =  out[i][j];
			}
		}
		
		int inpl = input.length;
		double finalcost = this.Costfunction(outp, changedy, inpl);
		//System.out.println(finalcost+ "final cost");
		this.predcost = finalcost;
		this.predcosts.add(finalcost);
		return outp;
		//ArrayList<Double> output = new ArrayList();
		//for (int i = 1; i < out.length; i++) {
		//	output.add(out[i]);
		//	System.out.println(out[i]);
		//}
		//double max = Collections.max(output);
		//System.out.println(max+ "this is max");
		//int index = output.indexOf(max);
		//System.out.println(this.trueY[index]);
	}
	
	
	public ArrayList<Double> Getpredcosts() {
		return this.predcosts;
	}
	
	public Double Getpredcost() {
		return this.predcost;
	}
	public void Propagateback(int ind) {
		//System.out.println("final start");		
		int i = this.layerlist.size() -1;
		Layer fin = this.layerlist.get(i);
		if (ind == 0) {
			fin.Initweightgrad();
		}
		fin.Propagateback();

		this.finalweight.set(i, fin.getweight());
		double[][] diff = fin.Getdiff();

		i -= 1;
		//System.out.println("final done");
		while(i > 0) {
			Layer hidden = this.layerlist.get(i);
			hidden.Setdiff(diff);
			if (ind == 0) {
				hidden.Initweightgrad();
			}
			hidden.Propagateback();
			this.finalweight.set(i, hidden.getweight());
			diff = hidden.Getdiff();
			i -= 1;
		}
		//System.out.println("middle done");
		Layer input = this.layerlist.get(0);
		input.Setdiff(diff);
		if (ind == 0) {
			input.Initweightgrad();
		}
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
	
	public double Costfunction(double[][] output, double[][] changedy, int inpl) {
		double costs = 0;
		for(int i=0; i< output.length; i++ ) {
			for( int j = 0; j< output[0].length; j++) {
				costs += Math.log(output[i][j]) * (-1) * changedy[i][j] -Math.log(1-output[i][j]) * (1 - changedy[i][j]);
			}
		}
		costs = costs/inpl + this.Getregulation();
		return costs;
		
	}

	
	public void Train() {
		int i  = 0;
		while(i < this.maxit) {
			//System.out.println("!!!!!!!!start train");
			//System.out.println(i+ " this is i hhhh");
			this.finaloutput = this.propogateforward();
			if(i == 0) {
				this.Propagateback(0);
			}else {
				this.Propagateback(1);
			}
			
			i += 1;
		}
		int j = 0;
		/*** while(j < this.finalweight.size()) {
			double[][] weight = this.finalweight.get(j);
			for(double[] row: weight) {
				for(double c : row) {
					System.out.print(c);
				}
				System.out.println(" ");
			}
			System.out.println("_________________");
			j += 1;
		}***/
		
		this.propogateforward();
		this.printdata(this.finaloutput);
		System.out.println("This is the output for train data");
	}
	
	public double[][] GetfinalOutput(){
		return this.finaloutput;
	}
	public static void main(String[] args) {
		double[][] X = new double[21][2];
		X[0][0] = 1.0;
		X[0][1] = 1.0;
		X[1][0] = 1.0;
		X[1][1] = 2.0;
		X[2][0] = 2.0;
		X[2][1] = 1.0;
		X[3][0] = 0.5;
		X[3][1] = 1.0; 
		X[4][0] = 2.0;
		X[4][1] = 0.5; 
		X[5][0] = 0.3;
		X[5][1] = 0.6; 
		X[6][0] = 0.2;
		X[6][1] = 0.7; 
		X[7][0] = 1.2;
		X[7][1] = 0.3; 
		
		X[8][0] = 2.2;
		X[8][1] = 1.3; 
		X[9][0] = 1.5;
		X[9][1] = 3.0; 
		X[10][0] = 3.0;
		X[10][1] = 1.0; 
		X[11][0] = 2.4;
		X[11][1] = 2.0; 
		X[12][0] = 2.0;
		X[12][1] = 2.0; 
		X[13][0] = 3.5;
		X[13][1] = 10; 
		X[14][0] = 4.0;
		X[14][1] = 0.3; 
		X[15][0] = 2.0;
		X[15][1] = 3.0; 
		X[16][0] = 5.0;
		X[16][1] = 0.0;
		X[17][0] = 4.0;
		X[17][1] = 0.03;
		X[18][0] = 3.0;
		X[18][1] = 2.0;
		X[19][0] = 1.2;
		X[19][1] = 3.0;
		X[20][0] = 3.0;
		X[20][1] = 3.0;
		
		
		double[] Y = new double[21];
		Y[0] = 0;

		Y[1] = 0;

		Y[2] = 0;
		Y[3] = 0;
		Y[4] = 0;
		Y[5] = 0;
		Y[6] = 0;
		Y[7] = 0;
		Y[8] = 1;

		Y[9] = 1;
		Y[10] = 1;
		Y[11] = 1;

		Y[12] = 1;
		Y[13] = 1;
		Y[14] = 1;
		Y[15] = 1;
		Y[16] = 1;
		Y[17] = 1;
		Y[18] = 1;
		Y[19] = 1;
		Y[20] = 1;
		
		
		//Numhiddenlayer: 4, InputValues: X, TrueY: Y
		//Num Neuron: 3 , regulation rate: 1,  Maxit: 2000, LR: 0.05

		Nnet nn = new Nnet(4, X, Y,3,7,2000, 1);
		nn.Train();
		double[][] k = nn.GetfinalOutput();
		
		
		//for (double[] d : k) {
		//	for(double dd: d) {
		//		System.out.print(dd);
		//	}
		//	System.out.println(" ");
		//}
		
		double[][] XX = new double[7][2];
		XX[0][0] = 0.0;
		XX[0][1] = 1.0;
	
		XX[1][0] = 0.0;
		XX[1][1] = 2.0;
		XX[2][0] = 3.0;
		XX[2][1] = 0.0;
		XX[3][0] = 0.0;
		XX[3][1] = 2.5;
		XX[4][0] = 6.0;
		XX[4][1] = 0.0;
		XX[5][0] = 7.0;
		XX[5][1] = 2.0;
		XX[6][0] = 3.0;
		XX[6][1] = 2.0;
		
		double [] yy = new double[7];
		yy[0] = 0;
		yy[1] = 0;
		yy[2] = 0;
		yy[3] = 0;
		yy[4] = 1;
		yy[5] = 1;
		yy[6] = 1;
				
		double[][] finalprediction = nn.predict(XX,yy);
		
		nn.printdata(finalprediction);
		/***for (double[][] dd : nn.finalweight) {
			for (double[] d: dd) {
				for(double dddd: d) {
					System.out.print(dddd+" ");
				}
				System.out.println(" ");
			}
			System.out.println("__________");
		}***/
		

 	}
	
}
