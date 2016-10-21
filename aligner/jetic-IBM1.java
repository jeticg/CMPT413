import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;


public class IBM1 {

	String[][] bitext_fe; //Bitext of source-target French to English (f-e)

	HashMap<String, Integer> f_count = new HashMap<String, Integer>(); //source words count map
	HashMap<String, Integer> e_count = new HashMap<String, Integer>(); //target words count map
	HashMap<Pair<String, String>, Integer> fe_count = new HashMap<Pair<String, String>, Integer>();    //source-target count map: For each key pair (f,e) fe_count assigns the number of occurrences of this pair to its value

	/**
	 * sets the training bitext over which the training process is going to be done
	 * @param bitext_fe : training bitext
	 */
	public IBM1(String[][] bitext_fe){
		this.bitext_fe = bitext_fe;
	}

	/**
	 * Reading a parallel corpus
	 * @param fileName
	 * @param fileName2
	 * @param linesToRead : number of lines to read from the parallel corpus (for the purpose of debugging)
	 * @return
	 * @throws IOException
	 */
	public static String[][] readBitext(String fileName, String fileName2, int linesToRead) throws IOException{
		BufferedReader br = null;
		BufferedReader br2 = null;

		String[][] text = new String[linesToRead][2];

	    try {
	    	File fileDir = new File(fileName);

			br = new BufferedReader(new InputStreamReader(new FileInputStream(fileDir), "UTF8"));

			File fileDir2 = new File(fileName2);

			br2 = new BufferedReader(new InputStreamReader(new FileInputStream(fileDir2), "UTF8"));

	        String line = br.readLine();
	        String line2 = br2.readLine();

	        int cnt = 0;
	        while (line != null) {
	        	text[cnt][0] = line;
	        	text[cnt][1] = line2;
	        	cnt++;
	        	System.out.println(cnt);
	            line = br.readLine();
	            line2 = br2.readLine();
	            if (cnt == linesToRead){
	            	return text;
	            }
	        }
	    } finally {
	        br.close();
	        br2.close();
	    }
		return text;
	}

	public void initializeCountsWithoutSets(){
		for (String[] fewa : bitext_fe){
			String[] F = fewa[0].split(" ");
			String[] E = fewa[1].split(" ");

			//Setting f_count
			for (String f : F){
				if (f_count.containsKey(f))
					f_count.put(f, new Integer(f_count.get(f).intValue()+1));
				else
					f_count.put(f, new Integer(1));

				//Setting fe_count
				for (String e : E){
					Pair<String, String> fePair = new Pair<String, String>(f,e);
					//System.out.println("pair "+f + " "  + e);
					if (fe_count.containsKey(fePair))
						fe_count.put(fePair, new Integer(fe_count.get(fePair).intValue()+1));
					else
						fe_count.put(fePair, new Integer(1));
				}
			}
			//setting e_count
			for (String e : E)
				if (e_count.containsKey(e))
					e_count.put(e, new Integer(e_count.get(e).intValue()+1));
				else
					e_count.put(e, new Integer(1));
		}

	}

	public HashMap<Pair<String, String>, Double> EM_IBM1(){
		//Pair p = new Pair("a", "b");
		HashMap<Pair<String, String>, Double> t = new HashMap<Pair<String, String>, Double>();

		double initialValue = new Double(1.0/f_count.size());

		for (Pair<String, String> p: fe_count.keySet()){
			t.put(p, initialValue);
		}
		//System.out.println("check "+ t.get(new Pair("on","one")));
		Pair<String, String> sdPair = new Pair<String, String>("","");
		Pair<String, String> newPair1 = new Pair<String, String>("","");

		for (int i = 0; i < 5 ; i++){
			System.out.println("IBM iteration " + i);
			HashMap<Pair<String, String>, Double> c = new HashMap<Pair<String, String>, Double>();
			HashMap<String, Double> total = new HashMap<String, Double>();

			//System.out.println(bitext.length);

			for (String[] pair : bitext_fe){
				String[] S = pair[0].split(" ");
				String[] D = pair[1].split(" ");

				for (String s_i : S){
					double Z = 0;
					for (String d_j : D){

						//System.out.println("t is null"+ s_i + d_j);
						sdPair.setPair(s_i, d_j);
						Z += t.get(sdPair).doubleValue();
					}
					for (String d_j : D){
						newPair1.setPair(s_i,d_j);

						//c[(f,e)] += t(f,e)/Z
						if (c.containsKey(newPair1))
							c.put(newPair1, new Double(c.get(newPair1).doubleValue() + t.get(newPair1).doubleValue()/Z));
						else{
							//c[(f,e)] = t(f,e)/Z
							Pair<String, String> newPair = new Pair<String, String>(s_i,d_j);
							c.put(newPair, new Double(t.get(newPair).doubleValue()/Z));
						}
						if (total.containsKey(d_j))
							total.put(d_j, new Double(total.get(d_j).doubleValue() + t.get(newPair1).doubleValue()/Z));
						else
							total.put(d_j, new Double( t.get(newPair1).doubleValue()/Z));
					}
				}
			}
			for (Pair<String, String> sd : fe_count.keySet()){
				t.put(sd, c.get(sd).doubleValue()/total.get(sd.getRight()).doubleValue());
			}
		}

		return t;
	}

	public double tProbability(String f, String e, HashMap<Pair<String, String>, Double> t_table){
		Pair<String, String> fe = new Pair<String, String>(f,e);

		int V = 163303;
		/*if (t_ibm1_20k.containsKey(fe))
		return t_ibm1_20k.get(fe);
		else*/
		if (t_table.containsKey(fe))
			return t_table.get(fe);
		else
			return 1.0/V;
	}

	public ArrayList<String> decodeToFile(String[][] bitext, HashMap<Pair<String, String>, Double> t, String alignmentFile) throws IOException{
		//returns an ArrayList of alignments results of aligning through IBM1 source to target
		BufferedWriter alignment = new BufferedWriter(new FileWriter(alignmentFile));
		ArrayList<String> alignmentList = new ArrayList<String>();
		for (String[] pair : bitext){
			String[] S = pair[0].trim().split(" ");
			String[] D = pair[1].trim().split(" ");

			HashSet<Pair<Integer, Integer>> set = new HashSet<Pair<Integer, Integer>>();
			for (int i = 0; i < S.length; i++){
				double max_t = 0;
				int argmax = -1;
				for (int j = 0; j < D.length; j++){
					//Pair p = new Pair(S[i],D[j]);
					double t_SiDj = tProbability(S[i], D[j], t);
					if ( t_SiDj > max_t ){
						max_t = t_SiDj;
						argmax = j;
					}
				}

				set.add(new Pair<Integer, Integer>(i,argmax));
			}
			String line = "";
			for (Pair<Integer, Integer> ip : set){
				//System.out.print((ip.getLeft()+1) + "-" + (ip.getRight()+1) + " ");
				line += (ip.getLeft()) + "-" + (ip.getRight()) + " ";
			}
			alignment.write(line+"\n");
			alignmentList.add(line);
			//System.out.println();
		}
		alignment.close();
		return alignmentList;
	}

	public static void main(String[] args) throws IOException{
		int trainingSize = 100000;
		int testSize = 1000;
		String alignmentFileName = "alignment";

		String[][] trainBitext = IBM1.readBitext("data/europarl.de","data/europarl.en",trainingSize);
		String[][] testBitext = IBM1.readBitext("data/europarl.de","data/europarl.en",testSize);

		IBM1 ibm1 = new IBM1(trainBitext);

		System.out.println("length " + ibm1.bitext_fe.length);

		ibm1.initializeCountsWithoutSets();

		HashMap<Pair<String, String>, Double> t_fe = ibm1.EM_IBM1();

		ArrayList<String> ibmModelAlignment = ibm1.decodeToFile(testBitext, t_fe, alignmentFileName);

	}
}
