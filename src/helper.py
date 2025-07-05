import pandas as pd

def convert_to_minutes(ser):
	return (
		ser
		.str.split(" ", expand=True)
		.set_axis(["hour", "minute"], axis=1)
		.assign(
			hour=lambda df_: (
				df_
				.hour
				.str.replace("h", "")
				.astype(int)
				.mul(60)
			),
			minute=lambda df_: (
				df_
				.minute
				.str.replace("m", "")
				.fillna("0")
				.astype(int)
			)
		)
		.sum(axis=1)
	)


def clean_data(df):
	return (
		df
		.drop(index=[6474])
		.drop_duplicates()
		.assign(**{
			col: df[col].str.strip()
			for col in df.select_dtypes(include="O").columns
		})
		.rename(columns=str.lower)
		.assign(
			airline=lambda df_: (
				df_
				.airline
				.str.replace(" Premium economy", "")
				.str.replace(" Business", "")
				.str.title()
			),
			date_of_journey=lambda df_: pd.to_datetime(df_.date_of_journey, dayfirst=True),
			dep_time=lambda df_: pd.to_datetime(df_.dep_time).dt.time,
			arrival_time=lambda df_: pd.to_datetime(df_.arrival_time).dt.time,
			duration=lambda df_: df_.duration.pipe(convert_to_minutes),
			total_stops=lambda df_: (
				df_
				.total_stops
				.replace("non-stop", "0")
				.str.replace(" stops?", "", regex=True)
				.pipe(lambda ser: pd.to_numeric(ser))
			),
			additional_info=lambda df_: df_.additional_info.replace("No info", "No Info")
		)
		.drop(columns="route")
	)