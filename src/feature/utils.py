import numpy as np
import joblib
import pandas as pd

import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
	OneHotEncoder,
	OrdinalEncoder,
	StandardScaler,
	MinMaxScaler,
	PowerTransformer,
	FunctionTransformer
)

from feature_engine.outliers import Winsorizer
from feature_engine.datetime import DatetimeFeatures
from feature_engine.selection import SelectBySingleFeaturePerformance
from feature_engine.encoding import (
	RareLabelEncoder,
	MeanEncoder,
	CountFrequencyEncoder
)

import matplotlib.pyplot as plt

import warnings


warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
sklearn.set_config(transform_output="pandas")

def select_duration(X):
    return X[["duration"]]

def is_direct(X):
	return X.assign(is_direct_flight=X.total_stops.eq(0).astype(int))

def have_info(X):
	return X.assign(additional_info=X.additional_info.ne("No Info").astype(int))

def is_north(X):
	columns = X.columns.to_list()
	north_cities = ["Delhi", "Kolkata", "Mumbai", "New Delhi"]
	return (
		X
		.assign(**{
			f"{col}_is_north": X.loc[:, col].isin(north_cities).astype(int)
			for col in columns
		})
		.drop(columns=columns)
	)

def part_of_day(X, morning=4, noon=12, eve=16, night=20):
	columns = X.columns.to_list()
	X_temp = X.assign(**{
		col: pd.to_datetime(X.loc[:, col]).dt.hour
		for col in columns
	})

	return (
		X_temp
		.assign(**{
			f"{col}_part_of_day": np.select(
				[X_temp.loc[:, col].between(morning, noon, inclusive="left"),
				 X_temp.loc[:, col].between(noon, eve, inclusive="left"),
				 X_temp.loc[:, col].between(eve, night, inclusive="left")],
				["morning", "afternoon", "evening"],
				default="night"
			)
			for col in columns
		})
		.drop(columns=columns)
	)

class RBFPercentileSimilarity(BaseEstimator, TransformerMixin):
	def __init__(self, variables=None, percentiles=[0.25, 0.5, 0.75], gamma=0.1):
		self.variables = variables
		self.percentiles = percentiles
		self.gamma = gamma


	def fit(self, X, y=None):
		if not self.variables:
			self.variables = X.select_dtypes(include="number").columns.to_list()

		self.reference_values_ = {
			col: (
				X
				.loc[:, col]
				.quantile(self.percentiles)
				.values
				.reshape(-1, 1)
			)
			for col in self.variables
		}

		return self


	def transform(self, X):
		objects = []
		for col in self.variables:
			columns = [f"{col}_rbf_{int(percentile * 100)}" for percentile in self.percentiles]
			obj = pd.DataFrame(
				data=rbf_kernel(X.loc[:, [col]], Y=self.reference_values_[col], gamma=self.gamma),
				columns=columns
			)
			objects.append(obj)
		return pd.concat(objects, axis=1)

def duration_category(X, short=180, med=400):
	return (
		X
		.assign(duration_cat=np.select([X.duration.lt(short),
									    X.duration.between(short, med, inclusive="left")],
									   ["short", "medium"],
									   default="long"))
		.drop(columns="duration")
	)

def is_over(X, value=1000):
	return (
		X
		.assign(**{
			f"duration_over_{value}": X.duration.ge(value).astype(int)
		})
		.drop(columns="duration")
	)

def make_air_transformer():
    air_transformer = Pipeline(steps=[
	("imputer", SimpleImputer(strategy="most_frequent")),
	("grouper", RareLabelEncoder(tol=0.1, replace_with="Other", n_categories=2)),
	("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ])
    return air_transformer


def make_doj_transformer():
    feature_to_extract = ["month", "week", "day_of_week", "day_of_year"]
    doj_transformer = Pipeline(steps=[
	("dt", DatetimeFeatures(features_to_extract=feature_to_extract, yearfirst=True, format="mixed")),
	("scaler", MinMaxScaler())
    ])
    return doj_transformer

def make_location_transformer():
    location_pipe1 = Pipeline(steps=[
	("grouper", RareLabelEncoder(tol=0.1, replace_with="Other", n_categories=2)),
	("encoder", MeanEncoder()),
	("scaler", PowerTransformer())
    ])
    location_transformer = FeatureUnion(transformer_list=[
	("part1", location_pipe1),
	("part2", FunctionTransformer(func=is_north))
    ])
    return location_transformer


def time_transformer():    
    time_pipe1 = Pipeline(steps=[
	("dt", DatetimeFeatures(features_to_extract=["hour", "minute"])),
	("scaler", MinMaxScaler())])
     
    time_pipe2 = Pipeline(steps=[
	("part", FunctionTransformer(func=part_of_day)),
	("encoder", CountFrequencyEncoder()),
	("scaler", MinMaxScaler())])

    time_transformer = FeatureUnion(transformer_list=[
	("part1", time_pipe1),
	("part2", time_pipe2)
    ])
    return time_transformer


def make_duration_tranformer():
    duration_pipe1 = Pipeline(steps=[
        ("rbf", RBFPercentileSimilarity()),
        ("scaler", PowerTransformer())
    ])
    duration_pipe2 = Pipeline(steps=[
        ("cat", FunctionTransformer(func=duration_category, validate=False)),  
        ("encoder", OrdinalEncoder(categories=[["short", "medium", "long"]]))
    ])

    part3 = Pipeline(steps=[
        ("is_over", FunctionTransformer(func=is_over, validate=False))  
    ])
    part4 = Pipeline(steps=[
        ("select_duration", FunctionTransformer(select_duration, validate=False)),
        ("scaler", StandardScaler())
    ])
    duration_union = FeatureUnion(transformer_list=[
        ("part1", duration_pipe1),
        ("part2", duration_pipe2),
        ("part3", part3),
        ("part4", part4)
    ])
    duration_transformer = Pipeline(steps=[
        ("outliers", Winsorizer(capping_method="iqr", fold=1.5, variables=["duration"])),
        ("imputer", SimpleImputer(strategy="median")),
        ("union", duration_union)
    ])

    return duration_transformer

def make_total_stops_tranaformers():
	total_stops_transformer = Pipeline(steps=[
	("imputer", SimpleImputer(strategy="most_frequent")),
	("", FunctionTransformer(func=is_direct))
	])
	return total_stops_transformer


def make_info_transformer():
    info_pipe1 = Pipeline(steps=[
        ("group", RareLabelEncoder(tol=0.1, n_categories=2, replace_with="Other")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    info_union = FeatureUnion(transformer_list=[
        ("part1", info_pipe1),
        ("part2", FunctionTransformer(func=have_info, validate=False))
    ])

    info_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("union", info_union)
    ])

    return info_transformer

def make_column_transformer():
    air = make_air_transformer()
    doj = make_doj_transformer()
    location = make_location_transformer()
    time = time_transformer()
    duration = make_duration_tranformer()
    stops = make_total_stops_tranaformers()
    info = make_info_transformer()

    column_transformer = ColumnTransformer(
        transformers=[
            ("air", air, ["airline"]),
            ("doj", doj, ["date_of_journey"]),
            ("location", location, ["source", "destination"]),
            ("time", time, ["dep_time", "arrival_time"]),
            ("dur", duration, ["duration"]),
            ("stops", stops, ["total_stops"]),
            ("info", info, ["additional_info"])
        ],
        remainder="passthrough"
    )

    return column_transformer

def make_selector():
	estimator = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
	selector = SelectBySingleFeaturePerformance(
	estimator=estimator,
	scoring="r2",
	threshold=0.1
	)
	return selector

def make_preprocessor():
	column_transformer = make_column_transformer()
	selector = make_selector()
	preprocessor = Pipeline(steps=[
	("ct", column_transformer),
	("selector", selector)
])
	return preprocessor 
	
def read_dataframe(path):
    df = pd.read_csv(path)
    return df

def save_dataframe(dataframe:pd.DataFrame, save_path):
    dataframe.to_csv(save_path,index=False)

def train_preprocessor(data : pd.DataFrame):
	preprocessor = make_preprocessor()
	y = data['price']
	X = data.drop(columns=['price'])
	preprocessor.fit(X,y)
	return preprocessor

def transform_data(transformer,data:pd.DataFrame):
    data_transformed = transformer.transform(data)
    
    return data_transformed

def save_transformer(path,object):
    joblib.dump(value=object,
                filename=path)

def read_dataframe(path):
    df = pd.read_csv(path)
    return df

def save_dataframe(dataframe:pd.DataFrame, save_path):
    dataframe.to_csv(save_path,index=False)

