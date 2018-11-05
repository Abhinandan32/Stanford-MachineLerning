package neuralnet;

import java.util.Scanner;

public class Driver {
	private Nnet serimod;
	
	public Driver() {
		
	}
	
	public int GetvalidInput(Scanner in) {
		int choice = in.nextInt();
		while(true) {
			if (choice != 1 && choice != 2) {
				System.out.println("Please choose a valid value");
				Scanner inn = new Scanner(System.in);
				choice = inn.nextInt();
			}else {
				break;
			}
		}
		return choice;
	}
	
	public static void main(String[] args) {
		System.out.println("Please decide which to start");
		System.out.println("1: start a new NN");
		System.out.println("Use an old model");
		
		Driver start = new Driver();
		Scanner in  = new Scanner(System.in);
		int choice = start.GetvalidInput(in);

	}
}
