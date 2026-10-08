# Error analysis

The project is structured to classify errors into the required categories:

- TP
- FP
- FN
- LOCALIZATION_ERROR

The analysis focuses on:

- clean-image false positives
- small-object false negatives
- low-contrast defects
- boundary defects
- multiple-instance defects
- domain-specific failures when labels are available

The script scaffold writes the required CSV fields and is ready to be connected to actual model predictions.
