package neuralnet;

import java.awt.Color;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import org.jfree.chart.*;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.block.BlockBorder;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.title.TextTitle;

public class LearningRate {
	private double[][] xt;
	private double[] yt;
	private double[][] x;
	private double[] y;
	private double[][] xcv;
	private double[] ycv;
	private double[] lambda;
	private ArrayList<Double> trainerror;
	private ArrayList<Double> cverror;
	private int split;
	private ArrayList<Integer> index;

	private XYSeriesCollection dataset;

	
	
	
	public LearningRate(double[][] X, double[] Y, double[] Lambda, int Split ) {
		this.xt = X;
		this.yt = Y;
		this.lambda = Lambda;
		this.trainerror = new ArrayList<Double>();
		this.cverror = new ArrayList<Double>();
		this.dataset = new XYSeriesCollection();
		this.split = Split;
	}
	
	public void Startshuffle() {
		this.index = new ArrayList<Integer>();
		for(int i = 0; i< this.xt.length; i++) {
			this.index.add(i);
		}
		Collections.shuffle(this.index);
		
	}
	
	public void StartSplit(int totn) {	
		int N = totn/5;		
		this.x = new double[3*N][this.xt[0].length];
		this.y = new double[3*N];

		int i = 0;
		while(i < 3*N) {
			this.x[i] = this.xt[index.get(i)];
			this.y[i] = this.yt[index.get(i)];
			i++;
		}
		
		this.xcv = new double[N][this.xt[0].length];
		this.ycv = new double[N];
		
		while(i< 4*N) {
			this.xcv[i-3*N] = this.xt[index.get(i)];
			this.ycv[i-3*N] = this.yt[index.get(i)];
			i++;
		}
		
		/***
		 * double[][] Xt = new double[N][X[0].length];
		 * 		double[] Yt = new double[N];
		
			while(i < 5*N) {
			Xt[i-4*N] = X[index.get(i)];
			Yt[i-4*N] = Y[index.get(i)];
			i++;
		}
		 */
	}
	
	public void Collecterrorlam() {
		for(int k = this.split; k < this.xt.length; k += this.split) {
			this.Startshuffle();
			this.StartSplit(k);
			//System.out.println("this is " + k);
			Nnet nn = new Nnet(10, this.x, this.y,3,0.01,100, 0.5);
			nn.Train();
			this.trainerror.add(nn.Getcost());
			nn.predict(this.xcv, this.ycv);
			this.cverror.add(nn.Getpredcost());
		}
	}
	
	public void addseries(ArrayList<Double> b, String c) {
		XYSeries series = new XYSeries(c);
		int N = this.xt.length / this.split;
		for(int i = 0; i < N; i++) {
			series.add(this.split*(i+1), b.get(i));
		}
		dataset.addSeries(series);
	}
	
	public void Visulization() {
		this.addseries(this.cverror, "cv");
		this.addseries(this.trainerror, "train");
		JFreeChart chart = ChartFactory.createXYLineChart("Learning Curve", "size", "J cost", this.dataset);
		XYPlot plot = chart.getXYPlot();
		XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
		renderer.setSeriesPaint(0, Color.RED);
		renderer.setSeriesPaint(1, Color.black);
		plot.setRenderer(renderer);
		chart.getLegend().setFrame(BlockBorder.NONE);
		chart.setTitle(new TextTitle("Learning rate for changing Lambda = 0.01"));
		try {
			ChartUtilities.saveChartAsPNG(new File("LR.png"), chart, 450, 400);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public ArrayList<Double> getTrainerror(){
		return this.trainerror;
	}
	
	public ArrayList<Double> getCVerror(){
		return this.cverror;
	}
	
	public static void main(String[] args) {
		
		String csvFile = "/Users/sjyuan/eclipse-workspace/MachineLearning/src/neuralnet/train.csv";
		BufferedReader br = null;
		String line = "";
		String cvsSplitBy = ",";
		
		ArrayList<double[]> all = new ArrayList<double[]>();
		
		try {
			br = new BufferedReader(new FileReader(csvFile));
			br.readLine();
			while ((line = br.readLine()) != null) {
				String[] row = line.split(cvsSplitBy);
				double[] newa = Arrays.stream(row).mapToDouble(Double::parseDouble).toArray();
				all.add(newa);
			}
		} catch(IOException e) {
			e.printStackTrace();
		}
		double[] n = all.get(0);
		double[][] X = new double[all.size()-1][n.length-1];
		double[] Y = new double[all.size()-1];
		//System.out.println(all.size()+ "size");
		for(int i = 1; i < all.size()-1; i++) {
			double[] arr = all.get(i);
			if(arr.length == all.get(0).length) {
				for(int j = 0; j < arr.length; j++) {
					
					
					if(j == 0) {
						
						Y[i] = arr[0];
					}else {
						//System.out.print("???");
						//System.out.print(arr[j]+ ",");
						X[i][j-1] = arr[j]; 
					}	
				}
				
			}

			//System.out.println(" ");
			//System.out.println(i+"+++++++++++");
		}
		//_____________________________________________________
		

		
		
		
		
		
		
		
		

	/***	
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
		***/
		
		double[] lambda = new double[6];
		lambda[0] = 0.05;
		lambda[1] = 0.1;
		lambda[2] = 0.5;
		lambda[3] = 1;
		lambda[4] = 2;
		lambda[5] = 3;
		
		System.out.println("start");
		LearningRate LR = new LearningRate(X, Y, lambda, 50);
		LR.Collecterrorlam();
		for( double dd: LR.cverror) {
			System.out.println(dd + " cv cost changed ");		}
		for(double dd: LR.trainerror) {
			System.out.println(dd+ " train cost changed");
		}
		LR.Visulization();
	}
}


