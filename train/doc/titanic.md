# 1 数据理解
##1.1 基本数据分析
数据量:  $891 \times 12$

字段 | 含义 | key | 缺失
---|---|---|---
PassengerId | 乘客Id |  | 0
Servived | 是否存活 | 0=死亡，1=存活 | 0
Pclass | 船舱登记 | 1(一等舱)，2(二等舱) 3(三等舱) | 0
Name | 名字 | | 0
Sex | 性别 |  | 0
Age | 年龄 | 各种年龄阶段 | 177
SibSp | 泰坦尼克号上兄弟/配偶的数目 |  | 0
Parch | 泰坦尼克号上父母/孩子的数目 |  | 0
Ticket | 票号 |  | 0
Fare | 乘客的票价 |  | 0
Cabin | 船舱的号码 |  | 697
Embarked | 乘船港口 | C=瑟堡， Q=皇后镇，S=南安普敦 | 2
![](https://raw.githubusercontent.com/xzyin/kaggle/master/train/image/train/titanic/1.0.PNG)

根据上述图片分析可以得到以下一些推论：

(1) 三等舱人数 > 一等舱 > 二等舱 (不过这个好像没什么用)

(2) 年龄在25岁左右的存活率达到最高大概在35%左右,存活率在2~3岁有一个比较异常的突起(这个凸起的原因可能需要进一步分析一下)。

(3) 随着舱位等级越高，人群的年龄越高，果然财富的积累需要时间?

(4) 大部分人是在南安普敦登船，竟然占据了75%左右，可能主要原因是因为这艘船在南安普顿出发的原因。想来当年杰克就是在这里赢了那张船票，开始了浪漫的没有归途的旅程吧! 可惜的是跟他一起赢了船票还没有跟妈妈告别的那两个倒霉蛋，果然交朋友还是要稳妥一点比较好。

![](https://raw.githubusercontent.com/xzyin/kaggle/master/train/image/train/titanic/1.1.PNG)
根据上面这两张图来看，可以在数据中发现：

(1) 舱位越高存活的可能性越高(果然是人穷命贱，还是好好挣钱)。

(2) 女人的存活率比男人要高很多。

根据上面的分析结果，可以分析一组组合特征的存活率情况:
![](https://raw.githubusercontent.com/xzyin/kaggle/master/train/image/train/titanic/1.2.PNG)

## 1.2 缺失数据处理方式
* 缺失的样本值占总数比例极高: 将数据舍弃，防止缺失数据过多反而引入噪声。
* 缺失值比例适中:

&emsp;&emsp;(1) 离散型数据:将确实的数据单独作为一个特征值加入进特征数据当中去。

&emsp;&emsp;(2) 连续型数据:考虑给定一个Step，将数据离散化，然后将缺失值作为一个单独的特征加入进去。
* 缺失数据不多: 试着使用已经有的数据，拟合数据，将数据补充上。
##
