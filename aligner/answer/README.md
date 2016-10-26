# Group Adam
We implemented both the IBM1 and the HMM model. And with the HMM model, we produced the best score(0.32) on the leaderboard. (Oct 22)

### How to run (align.py uses HMM model)
Note: Due to the fact that Python is a dynamically interpreted language, and training the HMM model involves a lot of nested loops(line 150~170, align.py), it takes about **7 hours** to train on the whole dataset. Please be patient while waiting.

     python answer/align.py -p europarl -f de > output.a
     head -1000 output.a > upload.a

## IBM1 model
The code is in method_IBM1.py

      for each (f, e):
        for each fi in f:
          find ej that maximize t(fi|ej)
          align fi to ej

It produced an alignment with a score of 0.438.

## HMM model
$$Pr(f_{1}^{J}, a_{1}^{J}|e_{1}^{I})$$
$$= Pr(J|e_{1}^{I})*\prod_{j=1}^{J}Pr(f_{j}, a_{j}|f_{1}^{j-1}, a_{1}^{j-1}, e_{1}^{I})$$
$$=Pr(J|e_{1}^{I})*\prod_{j=1}^{J}Pr(f_{j}|f_{1}^{j-1}, a_{1}^{j-1}, e_{1}^{I}) * Pr(a_{j}|f_{1}^{j-1}, a_{1}^{j-1}, e_{1}^{I})$$

Following the research [http://dl.acm.org/citation.cfm?id=778824](http://dl.acm.org/citation.cfm?id=778824), we trained a HMM model using the F\_count, Fe\_count, and translation table from IBM1 model.

Please note that the HMM model will need to use the IBM1 model class for initialisation.

It reaches performance of 0.32, currently the best score in leaderboard (Oct 22).
