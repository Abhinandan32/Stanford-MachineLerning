package neuralnet;

import java.util.Random;

public class InputLayer extends Layer{
	
	public InputLayer(double[][] Inputvalue, int Numoutput, double Lambda, double LR){
		super(Inputvalue, Numoutput, Lambda, LR);	
		Initweight();
	}
	
	public void Initweight() {
		super.Initweight();
		int m = this.Getnoutput();
		int n = this.Getinput()[0].length;
		this.Setweightinit(m, n+1);;

	}
	
	public void Propagateforward() {
		super.Propagateforward();
		double[][] inputvalue = this.Getinput();
		double[][] thisweight = this.getweight();
		int neuronnextlayer = thisweight.length;
		this.calculateProp(inputvalue, thisweight, neuronnextlayer);

		//System.out.println(thisweight+"weight"+" "+thisweight.length+" "+"Inputlayer");
	}
	
	public void Propagateback() {
		super.Propagateback();
		this.calcgrad(this.Getdiff(), this.Getinput());
	}

}
