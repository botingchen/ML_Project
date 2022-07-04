# TMDB-Box-Office Prediction
# Movie's Box Office Revenue Prediction
## I. Introduction
電影產業已經在人類社會中存在了長久的一段時間了。雖然不如股價、近年火紅的虛擬貨幣等常有爆炸性的成長或下跌，但它的產值也是正在逐年提升。電影儼然成為了人類生活娛樂中不可或缺的一部分。
在西元2018年，電影產業造就了約417億美元的收入，電影產業達到了前所未有的火熱。但是問題來了，究竟甚麼樣的因素，才能夠影響電影的票房收入呢？是電影的演員陣容呢？抑或是電影的成本、片長等？
因此在這份project，我決定透過電影的相關資訊，例如上映季節、製片公司、演員陣容等，來預估這支電影的預期票房收入區間範圍。除了能夠事先大概的估出票房外，也能夠透過一次更改一個特徵，如改變電影上映的季節，修改電影的片長等，來找出能夠獲取最大收益的條件，做出相對應的調整，最大化電影的票房。
## II. Data Collection
在收集資料的部分，我搜尋了一陣子，發現國外有一個好用的網站 [TMDB](https://www.themoviedb.org/?language=zh-TW)，裡面收集了各式各樣的電影相關資料，我只要申請並使用他們的api，就可以拿到需要的資料了。
申請帳號、拿到我的api key、研究了一番後，發現他的資料一頁只能容納20部電影，而且資料很分散，沒有整理起來，我必須透過不同的request才能夠收集所有我要的資料。所以我先用了兩層for迴圈，搭配下面的request，外層控制2016年到2021年，內層控制前50頁共6年 * 1000部電影 / 年，獲取組共6000部電影的基本資訊。
```
response = requests.get('https://api.themoviedb.org/3/discover/movie?api_key=c9d26a6782e25df032d62899e2bbe7ef&year='+ str(year) + '&page=' + str(page_num))
```

經過第一次的收集之後，我們有了一些基本資料，如TMDB賦予電影的id、上映日期、語言、類型等。但我發現我需要的演員陣容在另外一個地方。所以我將剛剛獲得的電影id存進一個list，再用for迴圈去跑，一部電影一部電影去查，就像下面的code一樣，得到每部電影的演員id、name、以及他們的popularity。
```
response = requests.get('https://api.themoviedb.org/3/movie/' + str(index) + '?api_key=c9d26a6782e25df032d62899e2bbe7ef&language=en-US')
```
最後我發現電影成本，以及我們的label，票房收益又在另一個地方，所以我一樣透過電影的id，搭配新的requst拿到剩下所有需要的資料。
```
response = requests.get('https://api.themoviedb.org/3/movie/' + str(index) + '?api_key=c9d26a6782e25df032d62899e2bbe7ef&language=en-US')
```
當然，request得到的資料會是json檔，也還有很多多餘的資料，所以還需要一連串的處理，才能得到我們最終的DataFrame，雖然繁瑣但技術含量不高，這邊就不提了。
## III. Preprocessing
1. Prepcessing的部分，我先是根據電影的Release Date，將他們劃分成春夏秋冬，四個季節，之後將分出來的季節以及電影的語言做label encode。
```
df['season_label_encode'] = labelencoder.fit_transform(df['season'])
df['original_language_label_encode'] = labelencoder.fit_transform(df['original_language'])
```
3. 再來，我開始著手處理missing data。先用`df.isnull().sum()`找出空白的featue，發現只有runtime有NULL，再用`df.fillna(int(df['runtime'].mean()), inplace=True)`將空白的那格填上平均的電影時長。
4. 我對演員陣容、電影類型、以及製片公司做one-hot。做到一半的時候發現演員陣容如果電影全部列出來的演員都做one hot的話，好像有點太多了，會到8萬個feature。我在做PCA跟直接丟掉data之間思考了一下，最後我決定根據TMDB提供的演員popularity，找出一部電影中popularity前三高，也就是最有名的那三個當作主要演員，成功濃縮到只剩1萬個。
```
genre_onehot = pd.Series(dataset['genre_name']).str.join('|').str.get_dummies()
company_onehot = pd.Series(dataset['company_name']).str.join('|').str.get_dummies()
cast_onehot = pd.Series(dataset['cast_name']).str.join('|').str.get_dummies()
dataset_onehot = pd.concat([dataset,genre_onehot,company_onehot,cast_onehot],axis = 1)
```
5. 最後的preprocessing是我做完之後才決定在加的。因為我原本是做數值的預測，但後來覺得這樣的預測結果太大膽了，我不如將收入分成數個區間，預測會在哪個區間。所以決定根據電影的收入分成ABCDEF六個區間，F代表虧錢，E是0 ~ 10000000，D是10000000 ~ 50000000，以此類推。
最後可以看到我們classification還算是蠻理想的。
![](https://i.imgur.com/05JsjLL.png)
![](https://i.imgur.com/LY8A3R0.png)

## IV. Data Visualization
我有做幾個簡單的Data Visualization，方便更加了解資料的分布或是相關性等。
### 1. 不同類型電影的票房
可以看到最賣座的前三名分別是adventure, action, fantasy。
![](https://i.imgur.com/DL8J4hZ.png)
### 2. 不同季節的電影票房
![](https://i.imgur.com/3GVd4Wg.png)
### 3. 電影成本分布
![](https://i.imgur.com/i1mH8ze.png)
還有很多其他的圖表，但我這邊就放幾個比較重要的就好。


## V. Models
Model的部分，我選擇先用Tree based的Decistion Tree以及Random Forest來比較一下雙方的performance。
### 1. Random Forest
將feature與label分離，feature存進X，label存進y。宣告Random Forest

`rf = RandomForestClassifier(n_estimators=100,bootstrap = True)`
validation我選擇用Kfold，K = 3下去跑。
Random Forest的參數我用grid search下去跑，最後發現n_estimators=100的效果最好，就保留下來了。

### 2. Decision Tree
將feature與label分離，feature存進X，label存進y。
`dstree = DecisionTreeClassifier(class_weight="balanced",max_depth = 40)`
validation我一樣選擇用Kfold，K = 3下去跑。
那Decision Tree的參數我也適用grid search，最後呈現出max_depth的performance最好，就留做使用了。
### 3. SVM
將feature與label分離，然後再用train test split將training set跟testing set分開。
validation我一樣選擇用Kfold，K = 3下去跑，但這次我是將剛剛分完的training set丟入，再分割成trainig set跟validation set。
那為了找到best hyperparameter，我將gamma用0.1,1,100帶入grid search去計算。
最後得到的結果如下圖
![](https://i.imgur.com/FSSUBHO.png)
所以最後用gamma = 1下去跑，將最一開始的步驟得到的training set(包括validation set)整個丟進去fit，然後用test set計算它的performance。

## VI. Results
### 這是Random Forest(n_estimators = 100)的performance
![](https://i.imgur.com/XfHm9En.png)

### 這是Decision Tree(class_weight="balanced,max_depth = 40")的performance
![](https://i.imgur.com/WDFDKaG.png)

### 這是SVM(kernel = "rbf",gamma = 1)的performance
![](https://i.imgur.com/56CL39M.png)




## VII. Conclusion
最終，我們如果透過Random Forest去預估，可以穩定的得到高達八成的準確率。在使用上，我們只要將電影中下列除了Revenue以及Revenue Level以外的所有資訊都丟進去我們的model，就能夠預估出他的Revenue Level。
![](https://i.imgur.com/lBkgdms.png)

舉例來說，假如我們今天要預測 "Harry Potter 20th Anniversary: Return to Hogwarts " 這部2022年上印的電影的票房，我們可以先透過TMDB 的 api抓下我們所有需要的資料，如下圖所示。
![](https://i.imgur.com/iQJZvIw.png)
之後我們再將這筆data跟我們所有的dataset做one hot以及label encode，轉換成最終我們需要的DataFrame內容。再丟入我們的model，就可以得到他預測出來的Revenue Level了。至於上映的季節，也可以透過更改data中 "Season"這個Feature，重新label encode，就能夠比較出predict出來的Revenue，選擇較適當的上映季節。
以我們的 " Harry Potter 20th Anniversary: Return to Hogwarts "為例，我們可以看到預測的結果，春天、秋天、以及冬天預測出來的票房都是A，夏天則是F，那電影的製片商就可以從剩下三個季節去做上映日期的選擇了。
![](https://i.imgur.com/R47tMsp.png)
當然除了季節以外，還有許多的Feature可以去做嘗試，但做法雷同，這邊就不再贅述了。
最終，我們得到我們想要的預測，完成了這項Project。


