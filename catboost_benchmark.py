from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score
from cnn import get_severity
import scanpy as sc

def main():
    '''
    Read in cellxgene matrix!

    :return: None
    '''
    # read in cell type data
    init_training = (sc.read_h5ad('training/CD4-positive, alpha-beta T cell_training'))
    init_testing = (sc.read_h5ad('testing/CD4-positive, alpha-beta T cell_testing'))

    training_labels = get_severity(init_training)
    testing_labels = get_severity(init_testing)

    training_dense = init_training.X.todense()
    testing_dense = init_testing.X.todense()
    
    # shape check:
    print(training_dense.shape)

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='RMSE',
        eval_metric='Accuracy',
        verbose=100,
        random_seed=42
    )

    train_pool=Pool(training_dense, training_labels)
    test_pool=Pool(testing_dense, testing_labels)

    model.fit(train_pool)

    preds = model.predict(test_pool)

    accuracy = accuracy_score(testing_labels, preds)
    print(f"Accuracy: {accuracy:.4f}")
