import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import re
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

#------------------------------------------------------------------
class mode_by_feature_imputer():
    """
    X_catの値ごとにX_naのmodeを計算し、その値でX_na内のnanを埋める
    X_na: fill値がある値のSeries
    X_cat: その値ごとのX_naの最頻値を算出するためのSeries
    """
    def __init__(self):
        pass
        
    def fit(self, X_na, X_cat):
        na_name = X_na.name
        cat_name = X_cat.name
        df = pd.DataFrame({na_name: X_na, cat_name: X_cat})
        self.dic = df.groupby(cat_name)[na_name].apply(lambda x: x.mode()[0]).to_dict()
        return self
        
    def transform(self, X_na, X_cat):
        ser = X_na.copy()
        ser[ser.isnull()] = X_cat[ser.isnull()].map(self.dic)
        return ser

def main():
    # ----------------------------------
    train = pd.read_csv("../data/raw/train.csv")
    test = pd.read_csv("../data/raw/test.csv")
    concat = [train, test]
    # ----------------------------------
    # null値が多すぎるのでdropするものたち
    drop_null_cols = ["PoolQC", "MiscFeature", "Alley", "Fence"]
    # train test に適用
    for df in concat:
        df.drop(columns=drop_null_cols, inplace=True)
    # ----------------------------------
    # List of 'NaN' including columns where NaN's mean none.
    # ここのfeaturesのcolumnのnanは値がないことを示している
    none_cols = [
        'FireplaceQu', 'GarageType',
        'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', "Exterior2nd"
    ]
    for df in concat:
        df[none_cols] = df[none_cols].fillna("None")
    # ----------------------------------
    # ここのcolumnsのnanは値が0であることを示している
    zero_cols = [
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
        'BsmtHalfBath', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea'
    ]
    for df in concat:
        df[zero_cols] = df[zero_cols].fillna(0)
    # ----------------------------------
    # そのほかのnull埋め
    # MSZoning, Exterior2nd, Exterior1st -> Neighborhood mode DataLeakage?
    # KitchenQual -> BuildMod mode DataLeakage?
    # Utilities Electrical Functional SaleType -> mode DataLeakge?
    # ----------------------------------
    mode_cols = ["Utilities", "Electrical", "Functional", "SaleType"]
    imputer = SimpleImputer(strategy="most_frequent")
    train[mode_cols] = imputer.fit_transform(train[mode_cols])
    test[mode_cols] = imputer.transform(test[mode_cols])
    # ----------------------------------
    # data lealage を抑えるためにtrainのみでBuildRemodYearごとのmodeを計算してそれをtrain, testに適用する
    my_imputer = mode_by_feature_imputer()
    my_imputer.fit(X_na=train["KitchenQual"], X_cat=train["YearRemodAdd"])
    train["KitchenQual"] = my_imputer.transform(train["KitchenQual"], train["YearRemodAdd"])
    test["KitchenQual"] = my_imputer.transform(test["KitchenQual"], test["YearRemodAdd"])
    # ----------------------------------
    # data lealage MSZoning, Exterior2nd, Exterior1stをtrainのみでNeighborhoodごとのmodeを計算してそれをtrain, testに適用する
    for col in ["MSZoning", "Exterior2nd", "Exterior1st", "LotFrontage"]:
        my_imputer = mode_by_feature_imputer()
        my_imputer.fit(X_na=train[col], X_cat=train["Neighborhood"])
        train[col] = my_imputer.transform(X_na=train[col], X_cat=train["Neighborhood"])
        test[col] = my_imputer.transform(test[col], test["Neighborhood"])
    # ----------------------------------
    #Utilities: most freqent value : 99.90 -> salepriceもこの値ではあまり変動ない -> いらない
    for df in concat:
        df.drop(columns=["Utilities"], inplace=True)
    # ----------------------------------
    #LandSlope: most freqent value : 95.17 -> salepriceの変動はあまりないが、関係はありそう -> Otherとして残す
    for df in concat:
        df["LandSlope"] = np.where(df["LandSlope"] == "Gtl", df["LandSlope"], "Other")
    # ----------------------------------
    # Condition2: most freqent value : 98.97 -> salepriceの変動が大きいがNormが多すぎる -> 全部残す
    # RoofMatl: most freqent value : 98.53 -> WdShake, WdShngl は高い傾向 CompShgこれが一番多いが
    def encode_RoofMatl(x):
        if x == "CompShg":
            ans = "CompShg"
        elif x in ["WdShake", "WdShake"]:
            ans = "Wood"
        else:
            ans = "Other"
        return ans

    for df in concat:
        df["RoofMatl"] = df["RoofMatl"].apply(encode_RoofMatl)
    # ----------------------------------
    # Heating: most freqent value : 98.46 -> Gasとそれ以外に分ければよいか 2値なら問題ない
    for df in concat:
        df["Heating"] = df["Heating"].apply(lambda x: "Gas" if re.match("Gas", x) else "Other")
    # ----------------------------------
    
    # order があるcategoricalの値
    ordinal_map = {'Ex': 5,'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None':0}
    fintype_map = {'GLQ': 6,'ALQ': 5,'BLQ': 4,'Rec': 3,'LwQ': 2,'Unf': 1, 'None': 0}
    expose_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0}

    ord_col = ['ExterQual','ExterCond','BsmtQual', 'BsmtCond','HeatingQC','KitchenQual','GarageQual','GarageCond', 'FireplaceQu']
    for df in concat:
        for col in ord_col:
            df[col] = df[col].map(ordinal_map)
        
    fin_col = ['BsmtFinType1','BsmtFinType2']
    for df in concat:
        for col in fin_col:
            df[col] = df[col].map(fintype_map)

    for df in concat:
        df['BsmtExposure'] = df['BsmtExposure'].map(expose_map)
    # ----------------------------------
    # 2値のものは0 1で変換 ordinal encoderでよい
    two_val_cols = ['Street', 'LandSlope', 'Heating', 'CentralAir']
    ordinal = OrdinalEncoder()
    train[two_val_cols] = ordinal.fit_transform(train[two_val_cols])
    test[two_val_cols] = ordinal.transform(test[two_val_cols])
    # ----------------------------------
    # 相関の強い連続値同士を落とす
    for df in concat:
        df.drop(columns=["Fireplaces", "GarageArea", "GarageQual", "GarageCond"], inplace=True)
    # ----------------------------------
    # 外れ値処理 features target
    count_features = ["LotFrontage", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF"]

    for df in concat:
        for col in count_features:
            ser = df[col]
            q_1, q_99 = np.percentile(ser, 1), np.percentile(ser, 99)
            df[col] = np.clip(ser, q_1, q_99)
    # -----------------------------------
    # 残りのcategoryの値をohe encoding
    col_features = [col for col in train.columns if train[col].dtypes == "object"]
    
    train_ohe = pd.get_dummies(train[col_features]).astype(int)
    train = pd.concat([train, train_ohe], axis=1)
    train.drop(columns=col_features, inplace=True)

    test_ohe = pd.get_dummies(test[col_features]).astype(int)
    test = pd.concat([test, test_ohe], axis=1)
    test.drop(columns=col_features, inplace=True)
    
    # 最後にtarget をlog変換する
    train["SalePrice"] = np.log(train["SalePrice"])

    return train, test

if __name__ == "__main__":
    train, test = main()

    train.to_csv("../data/processed/nb002_oheecoding_train.csv", index=False)
    test.to_csv("../data/processed/nb002_oheencoding_test.csv", index=False)
