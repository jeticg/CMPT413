import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;


public class HMM {

    String[][] bitext_fe;
    HashMap<String, Integer> f_count = new HashMap<String, Integer>();
    HashMap<Pair<String, String>, Integer> fe_count = new HashMap<Pair<String, String>, Integer>();
    HashSet<Integer> targetLengthSet;

    protected double[][][] a;

    protected HashMap<Pair<String, String>, Double> t_table;

    protected double[] pi;

    double p0H = 0.3;
    double nullEmissionProb = 0.000005;
    double smoothFactor = 0.1;

    /**
     * construct an HMM model with t_table generated from IBM1
     * @param t_table
     */
    public HMM(HashMap<Pair<String, String>, Double> t_table, String[][] bitext_fe, HashMap<String, Integer> f_count, HashMap<Pair<String, String>, Integer> fe_count) {
        this.t_table = t_table;
        this.bitext_fe = bitext_fe;
        this.f_count = f_count;
        this.fe_count = fe_count;
    }

    /**
     *
     * @param a : transition parameter
     * @param pi: initial parameter
     * @param y : source sentence
     * @param N : target sentence length
     * @param d : target sentence
     * @param t_table: translation parameter
     */
    public Pair<double[][], double[]> forwardWithTScaled(double[][][] a, double[] pi, String[] y, int N, int T, String[] d, HashMap<Pair<String, String>, Double> t_table){
        double[] c_scaled = new double[T+1];
        double[][] alphaHat = new double[N+1][];
        for (int i = 0; i < alphaHat.length ; i++){
            alphaHat[i] = new double[T+1];
        }
        double totalAlphaDoubleDot = 0;

        for (int i = 1; i < N+1 ; i++){
            alphaHat[i][1] = pi[i]*t_table.get(new Pair<String, String>(y[0], d[i-1])).doubleValue();
            totalAlphaDoubleDot += alphaHat[i][1];
        }

        c_scaled[1] = 1.0/totalAlphaDoubleDot;

        for (int i = 1; i < N+1 ; i++){
            alphaHat[i][1] = c_scaled[1]*alphaHat[i][1];
        }
        for (int t = 1; t < T; t++){
            totalAlphaDoubleDot = 0;
            for (int j = 1; j < N + 1 ; j++){
                double total = 0;
                for (int i = 1; i < N + 1 ; i++)
                    total += alphaHat[i][t]*a[i][j][N];
                alphaHat[j][t+1] = t_table.get(new Pair<String, String>(y[t],d[j-1]) )*total;
                totalAlphaDoubleDot += alphaHat[j][t+1];
            }
            c_scaled[t+1] = 1.0/totalAlphaDoubleDot;
            for (int i = 1; i < N + 1 ; i++)
                alphaHat[i][t+1] = c_scaled[t+1]*alphaHat[i][t+1];
        }
        return new Pair<double[][], double[]>(alphaHat, c_scaled);
    }
    /**
     *
     * @param a : transition parameter
     * @param pi: initial parameter
     * @param y : source sentence
     * @param N : target sentence length
     * @param T : source sentence length
     * @param d : target sentence
     * @param t_table: translation parameter
     * @param c_scaled: rescaler for HMM to avoid vanishing probabilities
     * @return
     */
    public double[][] backwardWithTScaled(double[][][] a, double[] pi, String[] y, int N, int T, String[] d, HashMap<Pair<String, String>, Double> t_table, double[] c_scaled){
        double[][] betaHat = new double[N+1][T+1];

        for (int i = 1; i < N+1 ; i++){
            betaHat[i][T] = c_scaled[T];
        }
        for (int t = T-1; t > 0; t--){
            for (int i = 1; i < N+1 ; i++){
                double total = 0;
                for (int j = 1; j < N + 1 ; j++){
                    total += betaHat[j][t+1]*a[i][j][N]*t_table.get(new Pair<String, String>(y[t], d[j-1]));
                }
                betaHat[i][t] = c_scaled[t]*total;
            }
        }
        return betaHat;
    }


    /**
     * computes the maximum target sentence length
     * @param bitext
     * @return
     */
    public Pair<Integer, HashSet<Integer>> maxTargetSentenceLength(String[][] bitext){
        int maxLength = 0;

        HashSet<Integer> targetLengthSet = new HashSet<Integer>();
        for (String[] fe : bitext){
            int tempLength = fe[1].split(" ").length;
            if ( tempLength > maxLength)
                maxLength = tempLength;
            targetLengthSet.add(tempLength);
        }
        return new Pair<Integer, HashSet<Integer>>(maxLength, targetLengthSet);
    }
    /**
     *
     * @param sd_count
     * @return
     */
    public Pair<HashMap<Pair<String, String>, Integer>, HashMap<Integer, Pair<String, String>>> mapBitextToInt(HashMap<Pair<String, String>, Integer>sd_count){
        HashMap<Pair<String, String>, Integer> index = new HashMap<Pair<String, String>, Integer>();
        HashMap<Integer, Pair<String, String>> biword = new HashMap<Integer, Pair<String, String>>();
        int i = 0;
        for (Pair<String, String> pair : sd_count.keySet()){
            index.put(pair, i);
            biword.put(i, pair);
            i++;
        }
        return new Pair<HashMap<Pair<String, String>, Integer>, HashMap<Integer, Pair<String, String>>>(index, biword);
    }
    /**
     * Initialize parameters
     * @param N : target sentence length
     */
    public void  initializeModel(int N){
        int twoN = 2*N;
        a = new double[twoN+1][twoN+1][N+1];

        pi = new double[twoN+1];
        for (int i = 1; i < twoN+1; i++)
            pi[i] = 1.0/twoN;
        for (int i = 1; i < N+1; i++)
            for (int j = 1; j < N + 1; j++)
                a[i][j][N] = 1.0/N;

    }
    /**
     *
     * @param tritext_sd tritext of train data
     * *
     * @param s_count
     * @param sd_count
     * @param count_feFileName
     * @throws IOException
     */
    public void baumWelch() {

        Pair<Integer, HashSet<Integer>> NandtargetLengthSet = maxTargetSentenceLength(bitext_fe);

        int N = NandtargetLengthSet.getLeft();
        targetLengthSet = NandtargetLengthSet.getRight();

        System.out.println("N "+N);
        Pair<HashMap<Pair<String, String>, Integer>, HashMap<Integer, Pair<String, String>>> indexBiwordPair = mapBitextToInt(fe_count);
        HashMap<Pair<String, String>, Integer> indexMap = indexBiwordPair.getLeft();
        HashMap<Integer, Pair<String, String>> biword = indexBiwordPair.getRight();

        int L = bitext_fe.length;
        int sd_size = indexMap.size();
        double[] totalGammaDeltaOverAllObservations_t_i = null;
        HashMap<String, Double>totalGammaDeltaOverAllObservations_t_overall_states_over_dest = null;

        for (int iteration = 0; iteration < 5; iteration++){

            double logLikelihood = 0;

            totalGammaDeltaOverAllObservations_t_i = new double[sd_size];
            totalGammaDeltaOverAllObservations_t_overall_states_over_dest = new HashMap<String, Double>();
            double[] totalGamma1OverAllObservations = new double[N+1];
            double[][][] totalC_j_Minus_iOverAllObservations = new double[N+1][N+1][N+1];
            double[][] totalC_l_Minus_iOverAllObservations = new double[N+1][N+1];

            long start0 = System.currentTimeMillis();

            for (String[] f_e : bitext_fe){

                String[] y = f_e[0].split(" ");
                String[] x = f_e[1].split(" ");
                //String[] wa = f_e_wa[2].split(" ");

                int T = y.length;
                N = x.length;
                HashMap<Integer, Double> c = new HashMap<Integer, Double>();

                if (iteration == 0)
                    initializeModel(N);

                Pair<double[][], double[]> alphaHatC = forwardWithTScaled(a, pi, y, N, T, x, t_table);

                double[][] alpha_hat = alphaHatC.getLeft();
                double[] c_scaled = alphaHatC.getRight();

                double[][] beta_hat = backwardWithTScaled(a, pi, y, N, T, x, t_table, c_scaled);

                double[][] gamma = new double[N+1][T+1];

                //Setting gamma
                for (int t = 1; t < T; t++){
                    logLikelihood += -1*Math.log(c_scaled[t]);
                    for (int i = 1; i < N+1; i++){
                        gamma[i][t] = (alpha_hat[i][t]*beta_hat[i][t])/c_scaled[t];
                        totalGammaDeltaOverAllObservations_t_i[indexMap.get(new Pair<String, String>(y[t-1], x[i-1])).intValue()] += gamma[i][t];
                    }
                }
                int t = T;
                logLikelihood += -1*Math.log(c_scaled[t]);
                for (int i = 1; i < N+1; i++){
                    gamma[i][t] = (alpha_hat[i][t]*beta_hat[i][t])/c_scaled[t];
                    totalGammaDeltaOverAllObservations_t_i[indexMap.get(new Pair<String, String>(y[t-1], x[i-1])).intValue()] += gamma[i][t];
                }


                for (t = 1; t < T; t++)
                    for (int i = 1; i < N+1; i++)
                        for (int j = 1; j < N+1; j++){
                            //currentPair.setPair(y[t],x[j-1]);
                            double xi = alpha_hat[i][t]*a[i][j][N]*t_table.get(new Pair<String, String>(y[t],x[j-1]))*beta_hat[j][t+1];
                            //double xi = alpha_hat[i][t]*a[i][j][N]*t_table.get(currentPair)*beta_hat[j][t+1];
                            if (c.containsKey(j-i)){
                                double value = c.get(j-i);
                                c.put(j-i, value+xi);
                            }else{
                                c.put(j-i, xi);
                            }
                        }

                for (int i = 1; i < N+1; i++){
                    for (int j = 1; j < N+1; j++)
                        totalC_j_Minus_iOverAllObservations[i][j][N] += (c.containsKey(j-i)? c.get(j-i) : 0);
                    for (int l = 1; l < N+1; l++)
                        totalC_l_Minus_iOverAllObservations[i][N] += (c.containsKey(l-i)? c.get(l-i) : 0);
                }


                for (int i = 1; i < N+1; i++)
                    totalGamma1OverAllObservations[i] += gamma[i][1];

            }//end of loop over bitext


            long start = System.currentTimeMillis();

            System.out.println("likelihood " + logLikelihood);
            N = totalGamma1OverAllObservations.length - 1;

            for (int k = 0; k < sd_size ; k++){
                totalGammaDeltaOverAllObservations_t_i[k] += totalGammaDeltaOverAllObservations_t_i[k];
                Pair<String, String> pair = biword.get(k);
                String e = pair.getRight();
                if (totalGammaDeltaOverAllObservations_t_overall_states_over_dest.containsKey(e)){
                    double value = totalGammaDeltaOverAllObservations_t_overall_states_over_dest.get(e);

                    totalGammaDeltaOverAllObservations_t_overall_states_over_dest.put(e, value + totalGammaDeltaOverAllObservations_t_i[k]);
                }else{

                    totalGammaDeltaOverAllObservations_t_overall_states_over_dest.put(e, totalGammaDeltaOverAllObservations_t_i[k]);
                }
            }

            long end = System.currentTimeMillis();
            System.out.println("time spent in the end of E-step: " + (end-start)/1000.0 );
            System.out.println("time spent in E-step: " + (end-start0)/1000.0 );

            int twoN = 2*N;

            //M-Step

            a = new double[twoN+1][twoN+1][N+1];
            pi = new double[twoN+1];
            t_table = new HashMap<Pair<String, String>, Double>();


            System.out.println("set " + targetLengthSet);
            for (int I : targetLengthSet){
                for (int i = 1; i < I+1; i++){
                    for (int j = 1; j < I+1; j++)
                        a[i][j][I] = totalC_j_Minus_iOverAllObservations[i][j][I]/totalC_l_Minus_iOverAllObservations[i][I];
                }
            }
            for (int i = 1; i < N+1; i++){
                //We can try uniform
                pi[i] = totalGamma1OverAllObservations[i]*(1.0/L);
            }

            for (int k = 0; k < sd_size ; k++){
                Pair<String, String> pair = biword.get(k);
                String f = pair.getLeft();
                String e = pair.getRight();
                t_table.put(new Pair<String, String>(f, e), totalGammaDeltaOverAllObservations_t_i[k]/totalGammaDeltaOverAllObservations_t_overall_states_over_dest.get(e));
            }

            long end2 = System.currentTimeMillis();
            System.out.println("time spent in M-step: " + (end2-end)/1000.0 );
            System.out.println("iteration " + iteration);
        }

    }
    /**
     *
     * @param p0H: probability of aligning to Null refer to Och and Ney, 2000a or 2003 (Systematic comparison of various statistical alignment models)
     */
    public void multiplyOneMinusP0H(){

        for (int I : targetLengthSet)
            for (int i = 1; i < I+1; i++)
                for (int j = 1; j < I+1; j++)
                    a[i][j][I] *= (1-p0H);
        for (int I : targetLengthSet){
            for (int i = 1; i < I+1; i++){
                for (int j = 1; j < I+1; j++){
                    //double temp = a[i][j][I];
                    //a[i][j][I] = (1-p0H)*temp;
                    a[i][i + I][I] = p0H;
                    a[i + I][i + I][I] = p0H;
                    a[i+I][j][I] = a[i][j][I];
                }
            }
        }
    }

    public double tProbability(String f, String e){
        Pair<String, String> fe = new Pair<String, String>(f,e);

        int V = 163303; //source side vocabulary
        if (t_table.containsKey(fe)){

            return t_table.get(fe);

        }
        else{
            if (e.equals("null"))
                return nullEmissionProb;
            else
                //System.out.println("smoothing t");
                return 1.0/V;
        }
            //return (count_fe.get(fe) + alpha)/ (count_e.get(e) + alpha*V);
    }
    //p(i|i',I) is smoothed to uniform distribution for now --> p(i|i',I) = 1/I
    //we can make it interpolation form like what Och and Ney did
    public double aProbability(int iPrime, int i, int I){
        if (targetLengthSet.contains(I))
            return a[iPrime][i][I];
        else{
            return 1.0/I;
        }
    }

    public ArrayList<Integer> logViterbi(int N, String[] o, String[] d){
        int twoN = 2*N;
        double[][] V = new double[twoN+1][o.length];
        int[][]ptr = new int[twoN+1][o.length];
        String[] newd = new String[2*d.length];
        int twoLend = 2*d.length;
        System.arraycopy(d, 0, newd, 0, d.length);
        for (int i = N; i < twoLend; i ++)
            newd[i] = "null";

        for (int q = 1 ; q < twoN + 1; q++){

                double t_o0_d_qMinus1 = tProbability(o[0],newd[q-1]);
                if (t_o0_d_qMinus1 == 0 || pi[q] == 0)
                    V[q][0] = Integer.MIN_VALUE;
                else
                    V[q][0] = Math.log(pi[q])+Math.log(t_o0_d_qMinus1);

        }

        //printArray(V);
        for (int t = 1; t < o.length; t++){
            for (int q = 1 ; q < twoN + 1; q++){
                double maximum = Integer.MIN_VALUE;
                int max_q = Integer.MIN_VALUE;
                double t_o_d_qMinus1 = tProbability(o[t], newd[q-1]);
                for (int q_prime = 1 ; q_prime < twoN + 1; q_prime++){
                    double a_q_prime_q_N = aProbability(q_prime, q, N);
                    if (a_q_prime_q_N != 0 && t_o_d_qMinus1 != 0){
                        double temp = V[q_prime][t-1]+Math.log(a_q_prime_q_N)+Math.log(t_o_d_qMinus1);
                        if ( temp > maximum){
                            maximum = temp;
                            max_q = q_prime;
                        }
                    }
                }
                V[q][t] = maximum;
                ptr[q][t] = max_q;
            }
        }

        double max_of_V = Integer.MIN_VALUE;
        int q_of_max_of_V = 0;
        for (int q = 1 ; q < twoN + 1; q++){
            if (V[q][o.length-1] > max_of_V){
                max_of_V = V[q][o.length-1];
                q_of_max_of_V = q;
            }
        }
        ArrayList<Integer> trace = new ArrayList<Integer>();
        trace.add(q_of_max_of_V);
        int q = q_of_max_of_V;
        int i = o.length-1;
        while (i > 0){
            q = ptr[q][i];
            trace.add(0, q);
            i = i - 1;
        }
        return trace;
    }

    public ArrayList<String> findBestAlignmentsForAll_AER(String[][] bitext, int num_lines, String alignmentFile) throws IOException{

        BufferedWriter alignment = new BufferedWriter(new FileWriter(alignmentFile));
        ArrayList<String> alignmentList = new ArrayList<String>();
        int n = 0;
        for (String[] pair : bitext){
            String[] S = pair[0].trim().split(" ");
            String[] D = pair[1].trim().split(" ");

            int N = D.length;
            ArrayList<Integer> bestAlignment = logViterbi(N, S, D);
            String line = "";
            for(int i = 0; i < bestAlignment.size(); i++){

                if (bestAlignment.get(i) <= N)
                    line += (i) +"-"+(bestAlignment.get(i)-1) +" ";
            }
            alignmentList.add(line);
            alignment.write(line+"\n");

            if (n == num_lines-1){
                alignment.close();
                return alignmentList;
            }
            n++;
        }
        alignment.close();
        return alignmentList;
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException{
        int trainingSize = 100000;
        int testSize = 1000;
        String alignmentFileName = "alignment";

        String[][] trainBitext = IBM1.readBitext("data/europarl.de","data/europarl.en",trainingSize);
        String[][] testBitext = IBM1.readBitext("data/europarl.de","data/europarl.en",testSize);

        IBM1 ibm1 = new IBM1(trainBitext);

        ibm1.initializeCountsWithoutSets();

        System.out.println("length " + trainBitext.length);

        HashMap<Pair<String, String>, Double> t_fe = ibm1.EM_IBM1();
        HMM hmmL = new HMM(t_fe, trainBitext,ibm1.f_count, ibm1.fe_count);
        hmmL.baumWelch();

        //hmmL.p0H = Double.parseDouble(args[0]);
        //hmmL.nullEmissionProb = Double.parseDouble(args[1]);
        hmmL.multiplyOneMinusP0H();

        System.out.println("p0H " + hmmL.p0H + " nullEmissionProb" + hmmL.nullEmissionProb);

        ArrayList<String> hmmModelAlignment = hmmL.findBestAlignmentsForAll_AER(testBitext, testSize, alignmentFileName);

        ArrayList<String> reference = ibm1.convertFileToArrayList("data/hansards.a");
        ibm1.gradeAlign(testSize, testBitext, reference, hmmModelAlignment);
    }

}
