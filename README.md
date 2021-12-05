# House Price Prediction

## Paper
| No | Titile | Detail | link | Status |
| --- | --- | --- | --- | --- |
| 1 | House Price EDA | EDAについてのkaggle notebook、きれいにまとまっている、categoricalの値をSalePriceの順位に応じて変換している | [url](https://www.kaggle.com/dgawlik/house-prices-eda) | Doing |
| 2 | Comprehensive data exploration with python | EDAについてのkaggle notebook、分布の正規性やoutlierの処理が参考になるか | [url](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python) | Doing |
| 3 | Top 1% Approach: EDA, New Models and Stacking | EDAについてのkaggle notebook ほとんどのnanは実際にnanではなく、ただ値がないということを示しているだけという知見が得られた | [url](https://www.kaggle.com/datafan07/top-1-approach-eda-new-models-and-stacking) | Doing |
| 4 | Data Science Workflow TOP 2% (with Tuning) | EDAからmodel作成までのwarkflow、univariable, bibariable(between features, target), categoryを変換するタイミング、その方法は確立されていない | [url](https://www.kaggle.com/angqx95/data-science-workflow-top-2-with-tuning) | Doing |
| 5 | LabelEncoder vs OneHotEncoding | tree base modelの場合ordinalenvoding で Linear model (NN, Logistic) はOneHotがいいとされている | [url](https://www.kaggle.com/general/166699) | Doing |
| 6 | Outlier!!! The Silent Killer | outlierの検出、処理に関する有益なnotebook | [url](https://www.kaggle.com/nareshbhat/outlier-the-silent-killer) | Doing |


## feature analysis
| Variable | Type | Segment | Expectation | Conclusion | Comment |



## Log
# 2021/12/03
* feature processing step
とりあえずnullの処理
1. ほとんどの値がnullのものを除く
    * train data の8割以上がnull PoolQC, MiscFeature, Alley, Fence
    * **Pool QC** pool sizeのcolumnがあるし、ほとんどnullなので一緒（一応pool qualityでsalepriceに差があるものの） これは落とす **pool ありなしはつくっていいかも**
    * **Miscfeatures** いかんせんこれもデータが少なすぎる落とす
    * **Alley** これも落とす
    * **Fence** この値はそれほどtargetに影響しない & nullの数が多い 落とす
    **結果 PoolQC, MiscFeature, Alley, Fence 全部落とす

2. nanがnullを示していないものがある
    * [Top 1% Approach: EDA, New Models and Stacking](https://www.kaggle.com/datafan07/top-1-approach-eda-new-models-and-stacking)を参考にして穴埋め
    * nanが本当に値がないものを示すもの（例えばGarageのないproperyのGarageQualityなど）や、nanが事実上0を示すものをNANという文字や0で埋める

3. そして残ったnull達
    * MSZoning 付近の地域の分類 <- これは**Neighborhoodからの情報で埋めればよいか**
    * LotFrontage 道までの距離 **保留** **catboostで埋めるか、modeで埋める** -> [Data Science Workflow TOP 2% (with Tuning)](https://www.kaggle.com/angqx95/data-science-workflow-top-2-with-tuning): **ここにあるようにNeighborhoodである程度変わるか**
    * NeighborhoodでLotFrontageもやる train はこれでいいけどtestにいれるのはけっこうだるいか **これだけちょっとconvertの方針が違う**
    * 残りは2とか1だけなのでmodeで埋めればよいか
    * Utilities 電気ガス水道が使えるか これもNeighborehoodによるか てかほとんど同じ値 -> **modeで十分**
    * Exterior1st, Exterior2nd これらのnullは同じrecordのもの **Neighborhoodごとに大きく変わる**
    * Electrical（電気）：mode(Neighbor)、KitchenQual（キッチンquality）: mode、Functional（保証とかそういうけい）:mode、SaleType（これないの意味わからんが）:mode **こういうの築年数andNeighborhoodに関係してそう**
    * Electrical Neighborhoodでわけても基本同じ値、old townだけちゃうものの割合が多い -> どちらかというと築年数が効いていそう **ほぼどこも同じ: **mode**
    * KitchenQual  Neighborhoodけっこう変わる、これも築年数効いてきそう **改築した年代による**
    * Functional どこも同じような値 どこも同じような値 **mode**
    * SaleType Neighborhoodごとには大体同じ 新しいほどnewは増えるが大体同じ **mode**
    

* heatmapで可視化して相関のあるfeature同士を省いたりする操作があるが、これはnumericalデータのみに限られる -> これの前にcategoricalの値を暫定的にでも変換しておくべきか

**[Data Science Workflow TOP 2% (with Tuning)](https://www.kaggle.com/angqx95/data-science-workflow-top-2-with-tuning) groupby transformの使い方 参考になる!!!!!!!**

* null 除くタイミング、categoryをconvertするタイミング -> categoryを軽く先に変換しても悪くなさそう！！！
* NAN含めてその値の分布を出すのはいいかもしれない！！[url](https://www.kaggle.com/dgawlik/house-prices-eda)

* rareな値をひとつにまとめたい、categoryをconvertしたい
    * rareな値に関してはbigなやつとそれ以外に分けたい
    * Street: most freqent value : 99.59 -> もともと2値 若干値によって差はある -> もともと2個なら残してもよいか
      Utilities: most freqent value : 99.90 -> salepriceもこの値ではあまり変動ない -> いらない
      LandSlope: most freqent value : 95.17 -> salepriceの変動はあまりないが、関係はありそう -> Otherとして残す
      Condition2: most freqent value : 98.97 -> salepriceの変動が大きいがNormが多すぎる -> 全部残す **target encoding?**
      RoofMatl: most freqent value : 98.53 -> WdShake, WdShngl は高い傾向 CompShgこれが一番多いが
      Heating: most freqent value : 98.46 -> Gasとそれ以外に分ければよいか 2値なら問題ない
    * order があるcategoricalの値から数値に変換する
    
    * 残りのやつらはOneHotかTargetになる **後で<----------**
        * uniqueなvalueが多すぎるものはtarget のほうがいいかも
        * ここは一括で NN用にはOneHot, tree model系にはOridinalで変換してやる

* 連続値と、ordinalのcategoricalの値同士の相関をheatmapで見る
    * 相関の高いもの同士は落とす?
    * FireplaceQu vs Fireplaces -> FireplaceQu でfireplace無いときNone入れてるのでFireplaceQuを残す
    * GarageCars vs GarageArea -> GarageCars残す 値が離散値となってよいか
    * GarageQual vs GarageYrBlt
    * GarageCond vs GarageYrBlt
    * -> こいつらはGarageYrBlt残せばいい
    * __つまりFireplaces, GarageArea, GarageQual, GarageCondと落とせばよい__
    
**numericなcolumnにはoutlierという問題もある**
* 外れ値を省く、見たnotebookはけっこう主観的なやりかたで好きではない
* log 変換する方法も悪くなさそう
* ただ、外れ値除きすぎて使えるdataが少なくなっては元も子もない
* isoforestなどの外れ値検出アルゴリズム
* シンプルに単変量でみたときのIQRとかで省くのもあり

* log変換も多くのnotebookで行われていた
* 外れ値検出：1%点, 99%点で区切ったほうがよい
* 外れ値の扱い：それほどrecord数が多いわけではないので埋めたほうがいい、最も近い値 1%, 99%のどっちか
* etc: dbscan(parameterの設定がいる), Isoforestも試してみたい (外れ値の扱いむずい、何の値を埋める?)

outlierについての方針
* 外れ値のあるだろう連続値の分布をlog変換 -> __target に関しては、逆変換で戻せるから__
* 1% 99% の%点にクリップ -> features __np.log1pは全てのデータ点に影響してしまうから__



    
