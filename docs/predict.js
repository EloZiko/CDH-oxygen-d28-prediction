// Prediction with the model exported by train.py (docs/model.js).
//
// A patient is an array of the four predictors, in the order of
// MODEL.predictors, with NaN for a missing value. The same code runs in the
// web page and in tests/test_web_model.py, which checks that it gives the
// same results as CatBoost in Python.
//
// CatBoost builds "oblivious" trees: all the nodes at the same depth of a tree
// ask the same question, "is predictor f above border b?". The answer to
// question k gives bit k of the number of the leaf reached.

function splitAnswers(model, tree, patient) {
  return tree.splits.map(([feature, border]) => {
    const value = patient[feature];
    if (Number.isNaN(value)) return model.nanAsTrue[feature];
    return Math.fround(value) > border; // CatBoost compares 32-bit floats
  });
}

function leafIndex(model, tree, patient) {
  const answers = splitAnswers(model, tree, patient);
  return answers.reduce((leaf, answer, k) => leaf + (answer ? 2 ** k : 0), 0);
}

// Model output on the log-odds scale: the sum of the leaves reached.
function rawScore(model, patient) {
  const total = model.trees.reduce(
    (sum, tree) => sum + tree.leafValues[leafIndex(model, tree, patient)], 0);
  return model.scale * total + model.bias;
}

// The tool reports a class only: a positive score (predicted probability
// above 0.5) means the model predicts oxygen dependency at day 28.
function isHighRisk(model, patient) {
  return rawScore(model, patient) > 0;
}

// Expected output of one tree when only the predictors in `known` take the
// patient's values. At a question on another predictor, both answers are
// averaged, weighted by the training patients behind them (leaf weights).
// As in CatBoost, the last question is the root of the tree.
function expectedTreeValue(model, tree, patient, known) {
  const answers = splitAnswers(model, tree, patient);

  // [expected value, weight] below question `level`; `leaf` holds the bits
  // already chosen by the questions above.
  function visit(level, leaf) {
    if (level < 0) return [tree.leafValues[leaf], tree.leafWeights[leaf]];
    const [value0, weight0] = visit(level - 1, leaf);
    const [value1, weight1] = visit(level - 1, leaf + 2 ** level);
    const weight = weight0 + weight1;
    if (known[tree.splits[level][0]]) return [answers[level] ? value1 : value0, weight];
    if (weight === 0) return [0, 0];
    return [(weight0 * value0 + weight1 * value1) / weight, weight];
  }

  return visit(tree.splits.length - 1, 0)[0];
}

// SHAP values: how much each predictor moves this patient's score away from
// the average score. With four predictors, the Shapley formula can be applied
// directly to the 16 subsets of predictors (bit f of a subset = predictor f).
function shapValues(model, patient) {
  const n = model.predictors.length;
  const expected = [];
  for (let subset = 0; subset < 2 ** n; subset++) {
    const known = model.predictors.map((_, f) => (subset & (1 << f)) !== 0);
    const total = model.trees.reduce(
      (sum, tree) => sum + expectedTreeValue(model, tree, patient, known), 0);
    expected.push(model.scale * total);
  }

  return model.predictors.map((_, f) => {
    let shap = 0;
    for (let subset = 0; subset < 2 ** n; subset++) {
      if (subset & (1 << f)) continue;
      const size = countBits(subset);
      const weight = factorial(size) * factorial(n - size - 1) / factorial(n);
      shap += weight * (expected[subset | (1 << f)] - expected[subset]);
    }
    return shap;
  });
}

function countBits(subset) {
  let count = 0;
  for (; subset > 0; subset >>= 1) count += subset & 1;
  return count;
}

function factorial(k) {
  return k <= 1 ? 1 : k * factorial(k - 1);
}
