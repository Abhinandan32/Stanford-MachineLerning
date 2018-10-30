package neuralnet;

import java.util.Random;



public class HiddenLayer extends Layer{
	
	public HiddenLayer( int NumoutputWeight, Layer prevLayer, int Index, int Numneuron, double Lambda,  double LR) {
		super(NumoutputWeight, prevLayer,Index, Numneuron, Lambda, LR);
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
		super.Propagateback();
		this.calcgrad(this.Getdiff(), this.Getprevoutput());
		this.calchiddendiff();
	}
}
