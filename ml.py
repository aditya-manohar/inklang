# ml.py
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import pickle
from utils import inkl_print
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dataset(dataset_name: str, path: str = None):
    logging.debug(f"Loading dataset: {dataset_name}, path: {path}")
    dataset_name = dataset_name.lower() if isinstance(dataset_name, str) else None
    supported_datasets = ["mnist", "fashion_mnist", "cifar10"]
    
    try:
        if dataset_name in supported_datasets:
            logging.debug(f"Loading built-in dataset '{dataset_name}'")
            if dataset_name == "mnist":
                (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
            elif dataset_name == "fashion_mnist":
                (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
            elif dataset_name == "cifar10":
                (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
            x_train = x_train.astype("float32") / 255.0
            x_test = x_test.astype("float32") / 255.0
            if x_train.ndim == 3:
                x_train = np.expand_dims(x_train, -1)
                x_test = np.expand_dims(x_test, -1)
            x = np.concatenate([x_train, x_test], axis=0)
            y = np.concatenate([y_train, y_test], axis=0)
            df = pd.DataFrame(x.reshape(x.shape[0], -1))
            df['label'] = y
            logging.debug(f"Loaded built-in dataset '{dataset_name}' rows={len(df)} cols={len(df.columns)}")
        elif path:
            full_path = os.path.join("Uploads", path)
            logging.debug(f"Loading dataset from file: {full_path}")
            if path.endswith('.csv'):
                df = pd.read_csv(full_path)
            elif path.endswith('.json'):
                df = pd.read_json(full_path)
            else:
                raise ValueError(f"[inklang] Unsupported file format: {path}")
            if 'label' not in df.columns and dataset_name != "KMeans":
                raise ValueError(f"[inklang] Dataset must have a 'label' column for supervised learning. Found columns: {list(df.columns)}")
            logging.debug(f"Loaded dataset from '{full_path}' rows={len(df)} cols={len(df.columns)}")
        elif isinstance(dataset_name, pd.DataFrame):
            df = dataset_name
            logging.debug(f"Using provided DataFrame: rows={len(df)} cols={len(df.columns)}")
        else:
            raise ValueError(f"[inklang] Dataset '{dataset_name}' not supported or path not provided")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset '{dataset_name}': {str(e)}")
        inkl_print(f"Error loading dataset '{dataset_name}': {str(e)}")
        raise

def train_model(model_name: str, dataset, model_type: str, epochs: int):
    try:
        logging.debug(f"Training model '{model_name}' with {model_type}, epochs={epochs}")
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError(f"[inklang] Dataset must be a DataFrame, got {type(dataset)}")
        
        is_clustering = model_type == "KMeans"
        X = dataset.drop('label', axis=1).values if 'label' in dataset.columns else dataset.values
        y = dataset['label'].values if 'label' in dataset.columns else None
        
        if not is_clustering and y is None:
            raise ValueError(f"[inklang] Dataset must have a 'label' column for {model_type}")
        
        from sklearn.model_selection import train_test_split
        if is_clustering:
            X_train, X_test = X, None
            y_train, y_test = None, None
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_type == "RandomForest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
        elif model_type == "SVM":
            model = SVC(random_state=42)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
        elif model_type == "LinearRegression":
            model = LinearRegression()
            model.fit(X_train, y_train)
            from sklearn.metrics import r2_score
            accuracy = r2_score(y_test, model.predict(X_test))
        elif model_type == "KMeans":
            model = KMeans(n_clusters=10, random_state=42)
            model.fit(X)
            from sklearn.metrics import silhouette_score
            accuracy = silhouette_score(X, model.labels_) if len(np.unique(model.labels_)) > 1 else 0.0
        else:
            model = keras.Sequential([
                layers.Flatten(input_shape=(X_train.shape[1],)),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(len(np.unique(y)) if y is not None else 10, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=epochs, batch_size=64, verbose=2)
            _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        metric_name = "silhouette_score" if model_type == "KMeans" else "R²" if model_type == "LinearRegression" else "accuracy"
        logging.debug(f"Trained model '{model_name}'. {metric_name}: {accuracy:.4f}")
        inkl_print(f"Trained model '{model_name}'. {metric_name}: {accuracy:.4f}")
        return model, accuracy
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        inkl_print(f"Error during training: {str(e)}")
        return None, 0.0

def predict_model(model, dataset):
    try:
        logging.debug("Starting prediction")
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError(f"[inklang] Dataset must be a DataFrame, got {type(dataset)}")
        X = dataset.drop('label', axis=1).values if 'label' in dataset.columns else dataset.values
        if isinstance(model, (RandomForestClassifier, SVC, xgb.XGBClassifier, LinearRegression, KMeans)):
            predictions = model.predict(X)
        else:
            predictions = model.predict(X)
            predictions = np.argmax(predictions, axis=1)
        logging.debug(f"Predictions: {predictions.tolist()[:10]}... (total {len(predictions)})")
        inkl_print(f"Predictions: {predictions.tolist()[:10]}... (total {len(predictions)})")
        return predictions
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        inkl_print(f"Error during prediction: {str(e)}")
        return []

def save_model(model, path):
    try:
        full_path = os.path.join("Uploads", path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'wb') as f:
            pickle.dump(model, f)
        logging.debug(f"Saved model to '{full_path}'")
        inkl_print(f"Saved model to '{full_path}'")
        return full_path
    except Exception as e:
        logging.error(f"Error saving model: {str(e)}")
        inkl_print(f"Error saving model: {str(e)}")
        raise

def load_model(path):
    try:
        full_path = os.path.join("Uploads", path)
        with open(full_path, 'rb') as f:
            model = pickle.load(f)
        logging.debug(f"Loaded model from '{full_path}'")
        inkl_print(f"Loaded model from '{full_path}'")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        inkl_print(f"Error loading model: {str(e)}")
        return None
    
def preprocess_dataset(dataset, operation=None, value=None):
    try:
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError(f"[inklang] Dataset must be a DataFrame, got {type(dataset)}")
        
        if operation is None:
            raise ValueError(f"[inklang] Invalid preprocess operation: {operation}")
        
        df = dataset.copy()
        
        if operation == "normalize":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, float(value) if value else 1.0))
            feature_cols = df.columns.drop('label') if 'label' in df.columns else df.columns
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
            logging.debug(f"Normalized dataset: {len(df)} rows, {len(feature_cols)} features")
        
        elif operation == "standardize":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            feature_cols = df.columns.drop('label') if 'label' in df.columns else df.columns
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
            logging.debug(f"Standardized dataset: {len(df)} rows, {len(feature_cols)} features")
        
        elif operation == "sample":
            if value is None or not 0 < float(value) <= 1.0:
                raise ValueError(f"[inklang] Sample ratio must be between 0 and 1, got {value}")
            df = df.sample(frac=float(value), random_state=42)
            logging.debug(f"Sampled dataset: {len(df)} rows (ratio={value})")
        
        elif operation == "dropna":
            if value is not None:
                raise ValueError(f"[inklang] dropna does not accept a value, got {value}")
            original_rows = len(df)
            df = df.dropna()
            logging.debug(f"Dropped NA values: {original_rows} -> {len(df)} rows")
        
        elif operation == "onehot":
            if value is not None:
                raise ValueError(f"[inklang] onehot does not accept a value, got {value}")
            feature_cols = df.columns.drop('label') if 'label' in df.columns else df.columns
            categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns
            if not categorical_cols.empty:
                df = pd.get_dummies(df, columns=categorical_cols, dtype=int)
                logging.debug(f"One-hot encoded {len(categorical_cols)} columns: {list(categorical_cols)}")
            else:
                logging.debug("No categorical columns found for one-hot encoding")
        
        else:
            raise ValueError(f"[inklang] Unknown preprocess operation: {operation}")
        
        return df
    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        inkl_print(f"Error during preprocessing: {str(e)}")
        raise