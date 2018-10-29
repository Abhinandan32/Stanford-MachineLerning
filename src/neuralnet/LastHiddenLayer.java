package neuralnet;

public class LastHiddenLayer extends Layer {
		
	public LastHiddenLayer(Layer prevLayer, int numoutput, double[][] TrueY, double Lambda,  double LR) {
		super(prevLayer, numoutput, TrueY, Lambda, LR);
		Initweight();
	}
	
	public void Initweight() {
		super.Initweight();
		int m = this.Getnoutput();
		int n = this.Getprevoutput()[0].length;
		this.Setweightinit(m, n+1);
	}
	

	public void Propagateforward() {
		super.Propagateforward();
		double[][] inputvalue = this.Getprevoutput();
		double[][] thisweight = this.getweight();
		//System.out.println(thisweight+"weight"+" "+thisweight.length+" "+"middlelayer"+thisweight[0].length);
		int numoutput = this.Getnoutput();
		this.calculateProp(inputvalue, thisweight, numoutput);
		
		//System.out.println(inputvalue+" "+ numoutputweight+"whats wrong");
	}
	
	public void Propagateback() {
		double[][] pred = this.Getoutput();
		double[][] truey = this.GetTrueY();
		this.calculatediff(pred, truey);
		double[][] diff = this.Getdiff();
		double[][] input = this.Getprevoutput();
		//System.out.println(diff.length +" " + diff[0].length + "diff");
		this.calcgrad(diff, input);
		//System.out.println("calcgrad done");
		this.calchiddendiff();
		//System.out.println("calcgrad done");
		
	}
	
	private void calculatediff(double[][] pred, double[][] truey ) {
		double[][] diff = new double[pred.length][pred[0].length];
		for( int i = 0 ; i< pred.length; i++) {
			for( int j = 0; j < pred[0].length; j++) {
				diff[i][j]= pred[i][j] - truey[i][j];
			}
		}
		this.Setdiff(diff);
	}
	

	
}
