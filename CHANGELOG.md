# Change log

## Changes based on peer review

### Provide further explanation of why f2-score was used

#### Peer feedback

The feedback that promped this changes was as follows:

"You used F2-score, which can feel unusual compared to more common metrics. It would help to briefly explain why F2 was chosen (e.g., prioritizing recall and reducing false negatives) and why it fit the problem better than general metrics like accuracy or F1."

#### Change

##### Before

"We chose to use f2-score because it is most important to not miss true positives."

##### After

"We chose to use f2-score. In our dataset is fairly unballanced with the positive class making up about 14% of the data. Furthermore, we deem it to be a much large issue to miss true positive than to miss-identify negatives. This is why we have chosen to use f2-score: f2-score places twice as much emphasis on recall compared to precision."
