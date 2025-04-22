from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier           # ← switched to Extra Trees
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from sklearn.metrics import accuracy_score

# 1. Prepare data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define model (Extra Trees instead of Random Forest)
model = ExtraTreesClassifier(random_state=42)

# 3. Define hyper‑parameter search space
param_space = {
    'n_estimators': Integer(50, 300),
    'max_depth': Integer(3, 20),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 20),
    'max_features': Categorical(['auto', 'sqrt', 'log2'])
}

# 4. Bayesian optimiser
opt = BayesSearchCV(
    estimator=model,
    search_spaces=param_space,
    n_iter=32,              # number of search iterations
    cv=5,                   # cross‑validation folds
    scoring='accuracy',     # evaluation metric
    n_jobs=-1,              # parallel jobs
    random_state=42,
    verbose=0
)

# 5. Execute search
opt.fit(X_train, y_train)

# 6. Report best parameters and score
print("Best hyper‑parameters:", opt.best_params_)
print("Best CV accuracy:", opt.best_score_)

# 7. Evaluate on the test set
y_pred = opt.best_estimator_.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
