# MIT 6.034 Lab 8: Support Vector Machines
# Written by 6.034 staff

from svm_data import *
from functools import reduce
import math


#### Part 1: Vector Math #######################################################

def dot_product(u, v):
    return sum([x[0] * x[1] for x in zip(u, v)])

def norm(v):
    return math.sqrt(sum(map((lambda x: x*x), v)))


#### Part 2: Using the SVM Boundary Equations ##################################

def positiveness(svm, point):
    """Computes the expression (w dot x + b) for the given Point x."""
    return dot_product(point.coords, svm.w) + svm.b

def classify(svm, point):
    """Uses the given SVM to classify a Point. Assume that the point's true
    classification is unknown.
    Returns +1 or -1, or 0 if point is on boundary."""
    val = positiveness(svm, point)
    if val > 0:
        return 1
    elif val < 0:
        return -1
    else:
        return 0

def margin_width(svm):
    """Calculate margin width based on the current boundary."""
    return 2 / norm(svm.w)

def check_gutter_constraint(svm):
    """Returns the set of training points that violate one or both conditions:
        * gutter constraint (positiveness == classification, for support vectors)
        * training points must not be between the gutters
    Assumes that the SVM has support vectors assigned."""
    res = set()
    for point in svm.training_points:
        pos = positiveness(svm, point)
        if abs(pos) < 1:
            res.add(point)
    for point in svm.support_vectors:
        cls = positiveness(svm, point)
        if cls != point.classification:
            res.add(point)
    return res


#### Part 3: Supportiveness ####################################################

def check_alpha_signs(svm):
    """Returns the set of training points that violate either condition:
        * all non-support-vector training points have alpha = 0
        * all support vectors have alpha > 0
    Assumes that the SVM has support vectors assigned, and that all training
    points have alpha values assigned."""
    res = set()
    for point in svm.training_points:
        if point not in svm.support_vectors and point.alpha != 0:
            res.add(point)
    for point in svm.support_vectors:
        if point.alpha <= 0:
            res.add(point)
    return res

def check_alpha_equations(svm):
    """Returns True if both Lagrange-multiplier equations are satisfied,
    otherwise False. Assumes that the SVM has support vectors assigned, and
    that all training points have alpha values assigned."""
    cond_sum = 0
    weights = [0 for _ in range(len(svm.w))]
    for point in svm.support_vectors:
        cls = classify(svm, point)
        cond_sum += cls*point.alpha
        tmp = [cls * point.alpha * i for i in point.coords]
        for i in range(len(weights)):
            weights[i] += tmp[i]
    return cond_sum == 0 and weights == svm.w


#### Part 4: Evaluating Accuracy ###############################################

def misclassified_training_points(svm):
    """Returns the set of training points that are classified incorrectly
    using the current decision boundary."""
    res = set()
    for point in svm.training_points:
        if point.classification != classify(svm, point):
            res.add(point)
    return res


#### Part 5: Training an SVM ###################################################

def update_svm_from_alphas(svm):
    """Given an SVM with training data and alpha values, use alpha values to
    update the SVM's support vectors, w, and b. Return the updated SVM."""
    weights = [0.0 for _ in range(len(svm.training_points[0].coords))]
    for point in svm.training_points:
        cls = point.classification
        if cls != 0:
            tmp = [float(cls) * point.alpha * i for i in point.coords]
            for i in range(len(weights)):
                weights[i] += tmp[i]

    support = []
    for point in svm.training_points:
        if point.alpha > 0:
            support.append(point)

    b_max = float('-inf')
    b_min = float('inf')
    for point in support:
        b = float(point.classification) - dot_product(weights, point.coords)
        b_max = max(b_max, b)
        b_min = min(b_min, b)
    b = (b_min+b_max) / 2.0
    svm.w = weights
    svm.support_vectors = support
    svm.b = b
    return svm


#### Part 6: Multiple Choice ###################################################

ANSWER_1 = 11
ANSWER_2 = 6
ANSWER_3 = 3
ANSWER_4 = 2

ANSWER_5 = ['A', 'D']
ANSWER_6 = ['A', 'B', 'D']
ANSWER_7 = ['A', 'B', 'D']
ANSWER_8 = []
ANSWER_9 = ['A', 'B', 'D']
ANSWER_10 = ['A', 'B', 'D']

ANSWER_11 = False
ANSWER_12 = True
ANSWER_13 = False
ANSWER_14 = False
ANSWER_15 = False
ANSWER_16 = True

ANSWER_17 = [1,3,6,8]
ANSWER_18 = [1,2,4,5,6,7,8]
ANSWER_19 = [1,2,4,5,6,7,8]

ANSWER_20 = 6


#### SURVEY ####################################################################

NAME = None
COLLABORATORS = None
HOW_MANY_HOURS_THIS_LAB_TOOK = None
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
