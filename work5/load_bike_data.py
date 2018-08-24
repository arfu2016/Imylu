"""
@Project   : Imylu
@Module    : load_bike_data.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/24/18 1:16 PM
@Desc      : 
"""
from utils import load_to_df
from work5.logger_setup import define_logger
logger = define_logger('work5.load_bike_data')

# scaled_features = {}


def transform_data():
    cnt_df = load_to_df('hour')
    fields_to_drop = ['instant', 'dteday', 'atemp', 'workingday',
                      'casual', 'registered']
    cnt_df = cnt_df.drop(fields_to_drop, axis=1)
    fields_to_convert = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
                         'weathersit']
    for field in fields_to_convert:
        # cnt_df[field] = cnt_df[field].astype('object')
        cnt_df[field] = cnt_df[field].astype('str')
    # logger.debug(cnt_df.dtypes)

    # quant_features = ['cnt', 'temp', 'hum', 'windspeed']
    # # Store scalings in a dictionary so we can convert back later
    # for each in quant_features:
    #     mean, std = cnt_df[each].mean(), cnt_df[each].std()
    #     scaled_features[each] = [mean, std]
    #     cnt_df.loc[:, each] = (cnt_df[each] - mean) / std
    return cnt_df


def load_bike_sharing_data():
    cnt_df = transform_data()
    # logger.debug(type(cnt_df.head().values.tolist()[0][0]))
    logger.debug(cnt_df.head().iloc[:, :-1])
    return cnt_df.iloc[:, :-1].values.tolist(), cnt_df.iloc[:, -1].values.tolist()


if __name__ == '__main__':
    load_bike_sharing_data()
