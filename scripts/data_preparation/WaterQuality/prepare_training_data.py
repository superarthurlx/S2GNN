import os
import sys
import pickle
import argparse

import numpy as np
import pandas as pd

import pdb
sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from basicts.data.utils import mkdir

DATASET_NAME = 'WaterQuality'
station = "datasets/raw_data/{0}/station.csv".format(DATASET_NAME)
rainfall = "datasets/raw_data/{0}/rainfall.csv".format(DATASET_NAME)
quality = "datasets/raw_data/{0}/quality".format(DATASET_NAME)
dump_path = "datasets/raw_data/{0}/{0}"
q_cols = ["Time", "KM", "TN", "TP", "Rain"]

dfs = pd.read_csv(station)
station_id = dfs.iloc[:,2].values
station_dict = dict(zip(dfs.iloc[:,1], dfs.iloc[:,2]))

def process_rainfall():
	'''
	处理降水量数据，对齐水质数据的时间刻度
	返回三个特征：4小时内总量，最大，平均
	'''
	dfr = pd.read_csv(rainfall, header=0, index_col=0)
	dfr.index = pd.to_datetime(dfr.index)
	dfr_col = [col for col in dfr.columns if col in station_id]
	dfr = dfr[dfr_col]
	# dfr_mean = dfr.resample('4H').mean()
	dfr_sum = dfr.resample('4H').sum()
	# dfr_max = dfr.resample('4H').max()
	# return [dfr_sum, dfr_mean, dfr_max]
	return [dfr_sum]

def process_quality(dfr):
	'''
	处理水质数据
	依次读取目录下水质数据csv，整合降水量数据，可能except的情况：
		水质数据缺少对应的降水量数据
		station表中查不到（无法对应）

	'''
	files = os.listdir(quality)
	quality_paths = [os.path.join(quality, f) for f in files]
	files_replaced = [station_dict.get(item[:-4], item[:-4]) for item in files]
	df_list = []
	station_list = []

	for path in quality_paths:

		try:
			df = pd.read_csv(path, index_col=0)
			df.index = pd.to_datetime(df.index)

			item = os.path.basename(path)[:-4]
			station_id = station_dict.get(item, item)
			df = pd.merge(df, dfr[0][station_id], how='left', left_index=True, right_index=True)

			df = df.reset_index()
			df.columns = q_cols
			df_list.append(df)
			station_list.append(station_id)
		except:
			print(station_id)
			pass

	return df_list, station_list

def dataframe_to_numpy(dfk, dfn, dfp, dfr, start_day, threshold=0.75):
	'''
	根据日期筛选数据，nan数量超过1-threshold后丢掉整列，默认是nan超过25%就不要了
	剩下的nan用插值填充
	'''
	dfk = dfk[dfk.index >= start_day]
	dfn = dfn[dfn.index >= start_day]
	dfp = dfp[dfp.index >= start_day]
	dfr = dfr[dfr.index >= start_day]

	dfk = dfk.dropna(thresh=len(dfk)*threshold, axis=1)
	dfn = dfn.dropna(thresh=len(dfn)*threshold, axis=1)
	dfp = dfp.dropna(thresh=len(dfp)*threshold, axis=1)
	dfr = dfr.dropna(thresh=len(dfr)*threshold, axis=1)

	stations = dfk.columns  
	time_index = dfk.index

	dfk = dfk.interpolate(method='linear', axis=1)
	dfn = dfn.interpolate(method='linear', axis=1)
	dfp = dfp.interpolate(method='linear', axis=1)
	dfr = dfr.interpolate(method='linear', axis=1)

	dfk = dfk.fillna(method='ffill')
	dfn = dfn.fillna(method='ffill')
	dfp = dfp.fillna(method='ffill')
	dfr = dfr.fillna(method='ffill')

	dfk = np.expand_dims(dfk.values, -1)
	dfn = np.expand_dims(dfn.values, -1)
	dfp = np.expand_dims(dfp.values, -1)
	dfr = np.expand_dims(dfr.values, -1)

	data = np.concatenate((dfk, dfn, dfp, dfr), axis=-1)

	return data, stations, time_index

def merge_dataframe(df_list, station_list):
	'''
	处理成array，污染物顺序：KM, TN, TP, Rain
	数据最后日期应该都是2023-12-31，起始日期不确定，所以处理成三个不同时间段的数据，每个数据包括的站点不同
	'''
	cols = ["KM", "TN", "TP"]
	dfk = pd.DataFrame() # KM
	dfn = pd.DataFrame() # TN
	dfp = pd.DataFrame() # TP
	dfr = pd.DataFrame() # rain

	for i, df in enumerate(df_list):
		df = df.set_index('Time')

		dfk = pd.merge(dfk, df['KM'], how='outer', left_index=True, right_index=True)
		dfk = dfk.rename({'KM':station_list[i]}, axis=1)

		dfn = pd.merge(dfn, df['TN'], how='outer', left_index=True, right_index=True)
		dfn = dfn.rename({'TN':station_list[i]}, axis=1)

		dfp = pd.merge(dfp, df['TP'], how='outer', left_index=True, right_index=True)
		dfp = dfp.rename({'TP':station_list[i]}, axis=1)

		dfr = pd.merge(dfr, df['Rain'], how='outer', left_index=True, right_index=True)
		dfr = dfr.rename({'Rain':station_list[i]}, axis=1)


	# for start_day in ['2023-01-01', '2022-01-01']:
	# 	data, stations, time_index = dataframe_to_numpy(dfk, dfn, dfp, dfr, pd.to_datetime(start_day))

	# 	if start_day == '2023-01-01':
	# 	 	path = dump_path.format(DATASET_NAME+'1Y')
	# 	 	mkdir(path)
	# 	 	np.savez(path, stations=stations, data=data, time_index=time_index)
	# 	 	print(path, data.shape)

	# 	elif start_day == '2022-01-01':
		 	
	# 	 	path = dump_path.format(DATASET_NAME+'2Y')
	# 	 	mkdir(path)
	# 	 	np.savez(path, stations=stations, data=data, time_index=time_index)
	# 	 	print(path, data.shape)

	start_day ='2022-01-01'
	data, stations, time_index = dataframe_to_numpy(dfk, dfn, dfp, dfr, pd.to_datetime(start_day))

	path = dump_path.format(DATASET_NAME)
	mkdir(path)
	np.savez(path, stations=stations, data=data, time_index=time_index)
	print(path, data.shape)


def main():
	df_list = process_rainfall()
	df_list, station_list = process_quality(df_list) 
	merge_dataframe(df_list, station_list)

if __name__ == '__main__':
	main()