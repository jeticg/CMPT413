# Group Adam
We implement PRO algorithm, and improve its performance by adding feature and average perceptron.


# Usage
**Please run under this folder directly**

Due to submit size limit, for data we only upload

1. `rich_train.nbest`  augmented from `train.nbest` by adding 3 features
2. `rich_test.nbest`   augmented from `test.nbest`  by adding 3 features
3. `bleu_precal`       pre-computed bleu score
4. `train.en`, `test.en`, `train.fr`, `test.fr`

Generate more feature: `python AddFeature.py -n data/rich_train.nbest -e data/train.en -f train.fr -r rich_train.nbest` (rich_best is the output)

Train:  `python learn.py -n data/rich_train.nbest -e data/train.en -f train.fr > train.weights`

Test :  `python  reranker.py -n data/rich_test.nbest -w train.weights < output`

Evalute : `python score-rerank.py < output`

# Methods we used
## Baseline

We implements the baseline algorithm, which reaches score of 23.79 in test.

Usage: `python learn.py -n 'train.nbest' -e 'english' -f 'french'`

```python
for s1, s2 in samples:
  if s1.feature * theta < s2.feature * theta:
    theta += eta * (s1.feature - s2.feature)
    mistakes += 1
```

## Add feature

We add `length`, `untranslated word`, `IBM1 score` into nbest features. `python AddFeature.py -e English -f French -n nbest -r new_nbest`. And score boosts from 23.79 to 24.89.

## Average perceptron

We modify the perceptron to average perceptron,

```python
for s1, s2 in samples:
  if s1.feature * theta < s2.feature * theta:
    delta += eta * (s1.feature - s2.feature)
    mistakes += 1
theta += delta
```


# Methods we tried (not used in final version)
## Different degrees of feature

For features [f1, f2 .. fn], we tried to add degree 2 [f1^2, f2^2 ... fn^2]. It is thought give model more learning ability, but doesn't show a improvement in result.

## Normalize Feature

We normalized all feature by [standard deviation](http://www.d.umn.edu/~deoka001/Normalization.html) (ComputeMean.py), but score is even worse compare to original version.

## Momentum update

Since this update can be taken as a kind of gradient descent, we tried momentum update .

```python
pref_grad = 0
for s1, s2 in samples:
  if s1.feature * theta < s2.feature * theta:
    grad =  eta * (s1.feature - s2.feature)
    pred_grad = pred_grad * discount + grad
    theta += pred_grad
    mistakes += 1
```

It is usually useful in deep learning frameworks, but again, score doesn't change much.
