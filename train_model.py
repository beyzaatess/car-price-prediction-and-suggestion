import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    mean_absolute_percentage_error,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestRegressor
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def data_preprocessing(file_path='./cardekho.csv', current_year=2024):

    data = pd.read_csv(file_path)
    data = data.dropna()
    

    scaler = MinMaxScaler()
    data[['mileage(km/ltr/kg)', 'engine']] = scaler.fit_transform(data[['mileage(km/ltr/kg)', 'engine']])


    data['selling_price'] = np.log1p(data['selling_price'])


    data = pd.get_dummies(data, columns=['fuel', 'transmission', 'seller_type', 'owner'], drop_first=True)


    data['brand'] = data['name'].apply(lambda x: x.split()[0])
    data['model'] = data['name'].apply(lambda x: ' '.join(x.split()[1:]))

    data = pd.get_dummies(data, columns=['brand'], drop_first=True)


    vectorizer = TfidfVectorizer(max_features=50)
    tfidf_matrix = vectorizer.fit_transform(data['model'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    data = pd.concat([data.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)


    data['original_name'] = data['name']
    data['original_model'] = data['model']


    data = data.drop(['name', 'model'], axis=1)

    Q1 = data['selling_price'].quantile(0.25)
    Q3 = data['selling_price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data['selling_price'] >= lower_bound) & (data['selling_price'] <= upper_bound)]

    data['car_age'] = current_year - data['year']

    data = data.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtype == 'object' else col)
    data.fillna(data.median(), inplace=True)

    data.to_csv('cleaned_cardekho.csv', index=False)
    return data

def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='r2', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, label='Training Score')
    plt.plot(train_sizes, test_scores_mean, label='Validation Score')
    plt.title('Learning Curve')
    plt.xlabel('Training Examples')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.grid()
    plt.savefig('learning_curve.png')
    plt.show()

def train_and_save_model(data):
    X = data.drop(['selling_price'], axis=1)
    y = data['selling_price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=4,
        scoring='r2',
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Evaluate model performance
    y_pred = best_model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
        "Explained Variance": explained_variance_score(y_test, y_pred),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred)
    }

    print("Best Hyperparameters:", grid_search.best_params_)
    print("\nModel Performance (Test Set):")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
    feature_importances.sort_values(ascending=False).head(10).plot(kind='bar', figsize=(12, 6), color='skyblue')
    plt.title('Feature Importance Ranking')
    plt.ylabel('Importance')
    plt.xlabel('Features')
    plt.savefig('feature_importance.png')
    plt.show()

    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
    print("\nCross-Validation R2 Scores:", cv_scores)
    print("Mean R2 Score:", np.mean(cv_scores))

    plt.figure(figsize=(12, 10))
    sns.boxplot(data=cv_scores, color='cyan')
    plt.title('Cross-Validation R2 Score Distribution')
    plt.savefig('cv_score_distribution.png')
    plt.show()

    plt.figure(figsize=(12, 10))
    corr_matrix = X.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('feature_correlation.png')
    plt.show()

    error_distribution = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(error_distribution, kde=True, color='red')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.savefig('prediction_error_distribution.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_test, label='Actual', fill=True, alpha=0.5)
    sns.kdeplot(y_pred, label='Predicted', fill=True, alpha=0.5)
    plt.title('Price Distribution: Actual vs Predicted')
    plt.xlabel('Log Selling Price')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('price_distribution.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.savefig('true_vs_predicted.png')
    plt.show()

    y_test_class = (y_test > y_test.median()).astype(int)
    y_pred_class = (y_pred > y_test.median()).astype(int)
    cm = confusion_matrix(y_test_class, y_pred_class)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Below Median', 'Above Median'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)

    return best_model, X.columns.tolist()

if __name__ == "__main__":
    data = data_preprocessing(file_path='cardekho.csv')
    train_and_save_model(data)