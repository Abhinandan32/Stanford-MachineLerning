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
		int i = 0;
		for(double l: this.lambda) {
			Nnet nn = new Nnet(4, this.x, this.y,3,l,500, 0.05);
			nn.Start();
			this.trainerror.set(i, nn.Getcost());
			this.cverror.set(i, nn.testset(this.xcv, this.ycv));
			i += 1;
		}
	}
	
	public ArrayList<Double> getTrainerror(){
		return this.trainerror;
	}
	
	public ArrayList<Double> getCVerror(){
		return this.cverror;
	}
}


