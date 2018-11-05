package neuralnet;

import java.util.ArrayList;

public class LearningRate {
	private double[][] x;
	private double[] y;
	private double[][] xcv;
	private double[] ycv;
	private double[] lambda;
	private ArrayList<Double> trainerror;
	private ArrayList<Double> cverror;
	
	
	
	public LearningRate(double[][] X, double[] Y, double[][] Xcv, double[] Ycv, double[] Lambda) {
		this.x = X;
		this.y = Y;
		this.xcv = Xcv;
		this.ycv = Ycv;
		this.lambda = Lambda;
		this.trainerror = new ArrayList<Double>();
		this.cverror = new ArrayList<Double>();
	}
	
	public void Collecterrorlam() {
		for(int k = 0; k < this.lambda.length; k++) {
			System.out.println("this is " + k);
			Nnet nn = new Nnet(4, this.x, this.y,3,this.lambda[k],500, 0.05);
			nn.Start();
			this.trainerror.add(nn.Getcost());
			double cost = nn.predict(this.xcv, this.ycv);
			this.cverror.add(cost);
			
		}
	}
	
	public void Visulization() {
		
	}
	
	public ArrayList<Double> getTrainerror(){
		return this.trainerror;
	}
	
	public ArrayList<Double> getCVerror(){
		return this.cverror;
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

		
		double[][] Xcv = new double[7][2];
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
		
		
		double[] Ycv = new double[7];
		Y[0] = 1;

		Y[1] = 0;

		Y[2] = 0;
		Y[3] = 1;
		Y[4] = 1;
		Y[5] = 1;
		Y[6] = 1;
		
		
		double[] lambda = new double[6];
		lambda[0] = 0.05;
		lambda[1] = 0.1;
		lambda[2] = 0.5;
		lambda[3] = 1;
		lambda[4] = 2;
		lambda[5] = 3;
		
		System.out.println("start");
		LearningRate LR = new LearningRate(X, Y, Xcv, Ycv, lambda);
		LR.Collecterrorlam();
		for( double dd: LR.cverror) {
			System.out.println(dd + " cost changed ");		}
		
		
	}
}


