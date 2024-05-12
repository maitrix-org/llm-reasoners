# AQUA-RAT (Algebra Question Answering with Rationales) Dataset

This dataset contains the algebraic word problems with rationales described in our paper:

[Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. (2017) Program Induction by Rationale Generation: Learning to Solve and Explain Algebraic Word Problems. In Proc. ACL.](https://arxiv.org/pdf/1705.04146.pdf)

The dataset consists of about 100,000 algebraic word problems with natural language rationales. Each problem is a json object consisting of four parts:

 * `question` - A natural language definition of the problem to solve
 * `options` - 5 possible options (A, B, C, D and E), among which one is correct
 * `rationale` - A natural language description of the solution to the problem
 * `correct` - The correct option

Here is an example of a problem object:

    {
    "question": "A grocery sells a bag of ice for $1.25, and makes 20% profit. If it sells 500 bags of ice, how much total profit does it make?",
    "options": ["A)125", "B)150", "C)225", "D)250", "E)275"],
    "rationale": "Profit per bag = 1.25 * 0.20 = 0.25\nTotal profit = 500 * 0.25 = 125\nAnswer is A.",
    "correct": "A"
    }

## Files

 * `train.json` -> untokenized training set
 * `train.tok.json` -> tokenized training set
 * `dev.json` -> untokenized development set
 * `dev.tok.json` -> tokenized development set
 * `test.json` -> untokenized test set
 * `test.tok.json` -> tokenized test set

## Note

This dataset has been fully crowdsourced, as described using the technique in the paper (Ling et al., 2017). The initial published results included in the paper were derived from a previous version of this dataset that cannot be released in full, and results using the published system will differ. Results using our published system will be forthcoming.

