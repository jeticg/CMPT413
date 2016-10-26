# Group Adam
We implement both IMB1 and HMM model. And with HMM model, we produce the best score(0.32) in leaderboard . (Oct 22)

### How to run (align.py uses HMM model)
Note: Due to Python is a dynamic interpreted language, and training HMM model involves a lot of nested loops(line 150~170, align.py), it takes about **6 hours** to train on whole dataset. Please be patient while waiting.

    python answers/align.py -n 100000 -d data

## IBM1 model
Code is in model_IMB1.py

      for each (f, e):
        for each fi in f:
          find ej that maximize t(fi|ej)
          align fi to ej

It reaches performance of 0.438.

## HMM model
$$Pr(f_{1}^{J}, a_{1}^{J}|e_{1}^{I})$$
$$= Pr(J|e_{1}^{I})*\prod_{j=1}^{J}Pr(f_{j}, a_{j}|f_{1}^{j-1}, a_{1}^{j-1}, e_{1}^{I})$$
$$=Pr(J|e_{1}^{I})*\prod_{j=1}^{J}Pr(f_{j}|f_{1}^{j-1}, a_{1}^{j-1}, e_{1}^{I}) * Pr(a_{j}|f_{1}^{j-1}, a_{1}^{j-1}, e_{1}^{I})$$

Follow the research [http://dl.acm.org/citation.cfm?id=778824](http://dl.acm.org/citation.cfm?id=778824) we trained a HMM model using the F_count, Fe_count, optimised T from IMB1 model.

It reaches performance of 0.32, currently the best score in leaderboard (Oct 22).
