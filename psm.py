#处理data
import pandas as pd
df = pd.read_csv('psm.csv')
print(df.shape)
print(df.columns)
print(df.head())
print(df.describe())
#基本所有数据都要先看一下， max min mean sd
'''
(100, 4)
Index(['Too Cheap', 'Cheap', 'Expensive', 'Too Expensive'], dtype='object')
   Too Cheap  Cheap  Expensive  Too Expensive
0        100    150        400            500
1        120    200        400            480
2        200    250        450            500
3        200    300        350            400
4        300    340        400            490
        Too Cheap       Cheap  Expensive  Too Expensive
count  100.000000  100.000000  100.00000     100.000000
mean   218.250000  273.300000  344.05000     390.500000
std    110.793682  101.759322  107.18693     113.969028
min    100.000000  110.000000  130.00000     140.000000
25%    106.750000  197.500000  277.50000     300.000000
50%    200.000000  279.500000  384.50000     416.000000
75%    300.000000  352.500000  405.00000     499.250000
max    450.000000  460.000000  490.00000     500.000000
'''
#row and column 互换一下
print(df.unstack().reset_index())
'''
           level_0  level_1    0
0        Too Cheap        0  100
1        Too Cheap        1  120
2        Too Cheap        2  200
3        Too Cheap        3  200
4        Too Cheap        4  300
..             ...      ...  ...
395  Too Expensive       95  199
396  Too Expensive       96  410
397  Too Expensive       97  430
398  Too Expensive       98  280
399  Too Expensive       99  300
'''
df1 = (
       df
       .unstack()
       .reset_index().rename(columns={'level_0':'label', 0:'price'})
       .groupby(['label', 'price'])
       .size()
       .reset_index()
       .rename(columns={ 0:'freq'}))
'''
            label  price  freq
0           Cheap    110     5
1           Cheap    120     5
2           Cheap    129     5
3           Cheap    150     5
4           Cheap    190     5
5           Cheap    200    10
6           Cheap    240     5
7           Cheap    250     5
8           Cheap    260     5
9           Cheap    299     5
10          Cheap    300     5
11          Cheap    340     5
12          Cheap    350    10
13          Cheap    360    10
14          Cheap    388     5
15          Cheap    410     5
16          Cheap    460     5
17      Expensive    130     5
18      Expensive    149     5
19      Expensive    180     5
20      Expensive    200     5
21      Expensive    270     5
22      Expensive    280     5
23      Expensive    300     5
24      Expensive    350     5
25      Expensive    370    10
26      Expensive    399     5
27      Expensive    400    20
28      Expensive    420     5
29      Expensive    433     5
30      Expensive    450     5
31      Expensive    490    10
32      Too Cheap    100    25
33      Too Cheap    109    10
34      Too Cheap    120     5
35      Too Cheap    200    15
36      Too Cheap    250     5
37      Too Cheap    257     5
38      Too Cheap    280     5
39      Too Cheap    300    10
40      Too Cheap    340     5
41      Too Cheap    350     5
42      Too Cheap    400     5
43      Too Cheap    450     5
44  Too Expensive    140     5
45  Too Expensive    199     5
46  Too Expensive    200     5
47  Too Expensive    280     5
48  Too Expensive    300    10
49  Too Expensive    380    10
50  Too Expensive    400     5
51  Too Expensive    410     5
52  Too Expensive    422     5
53  Too Expensive    430     5
54  Too Expensive    480     5
55  Too Expensive    490     5
56  Too Expensive    499     5
57  Too Expensive    500    25
'''

df1['sum']=df1.groupby(['label'])['freq'].transform('sum')
df1['cumsum']=df1.groupby(['label'])['freq'].cumsum()
df1['percentage']=df1['cumsum']/df1['sum']*100
print(df1)
'''
           label  price  freq  sum  cumsum  percentage
0           Cheap    110     5  100       5         5.0
1           Cheap    120     5  100      10        10.0
2           Cheap    129     5  100      15        15.0
3           Cheap    150     5  100      20        20.0
4           Cheap    190     5  100      25        25.0
5           Cheap    200    10  100      35        35.0
6           Cheap    240     5  100      40        40.0
7           Cheap    250     5  100      45        45.0
8           Cheap    260     5  100      50        50.0
9           Cheap    299     5  100      55        55.0
10          Cheap    300     5  100      60        60.0
11          Cheap    340     5  100      65        65.0
12          Cheap    350    10  100      75        75.0
13          Cheap    360    10  100      85        85.0
14          Cheap    388     5  100      90        90.0
15          Cheap    410     5  100      95        95.0
16          Cheap    460     5  100     100       100.0
17      Expensive    130     5  100       5         5.0
18      Expensive    149     5  100      10        10.0
19      Expensive    180     5  100      15        15.0
20      Expensive    200     5  100      20        20.0
21      Expensive    270     5  100      25        25.0
22      Expensive    280     5  100      30        30.0
23      Expensive    300     5  100      35        35.0
24      Expensive    350     5  100      40        40.0
25      Expensive    370    10  100      50        50.0
26      Expensive    399     5  100      55        55.0
27      Expensive    400    20  100      75        75.0
28      Expensive    420     5  100      80        80.0
29      Expensive    433     5  100      85        85.0
30      Expensive    450     5  100      90        90.0
31      Expensive    490    10  100     100       100.0
32      Too Cheap    100    25  100      25        25.0
33      Too Cheap    109    10  100      35        35.0
34      Too Cheap    120     5  100      40        40.0
35      Too Cheap    200    15  100      55        55.0
36      Too Cheap    250     5  100      60        60.0
37      Too Cheap    257     5  100      65        65.0
38      Too Cheap    280     5  100      70        70.0
39      Too Cheap    300    10  100      80        80.0
40      Too Cheap    340     5  100      85        85.0
41      Too Cheap    350     5  100      90        90.0
42      Too Cheap    400     5  100      95        95.0
43      Too Cheap    450     5  100     100       100.0
44  Too Expensive    140     5  100       5         5.0
45  Too Expensive    199     5  100      10        10.0
46  Too Expensive    200     5  100      15        15.0
47  Too Expensive    280     5  100      20        20.0
48  Too Expensive    300    10  100      30        30.0
49  Too Expensive    380    10  100      40        40.0
50  Too Expensive    400     5  100      45        45.0
51  Too Expensive    410     5  100      50        50.0
52  Too Expensive    422     5  100      55        55.0
53  Too Expensive    430     5  100      60        60.0
54  Too Expensive    480     5  100      65        65.0
55  Too Expensive    490     5  100      70        70.0
56  Too Expensive    499     5  100      75        75.0
57  Too Expensive    500    25  100     100       100.0
'''
#pivot table
#value is percentage ,row: price ,column : cheap,expensive ,etc
df2 = df1.pivot_table('percentage','price','label')
print(df2)
'''
label  Cheap  Expensive  Too Cheap  Too Expensive
price                                            
100      NaN        NaN       25.0            NaN
109      NaN        NaN       35.0            NaN
110      5.0        NaN        NaN            NaN
120     10.0        NaN       40.0            NaN
129     15.0        NaN        NaN            NaN
130      NaN        5.0        NaN            NaN
140      NaN        NaN        NaN            5.0
149      NaN       10.0        NaN            NaN
150     20.0        NaN        NaN            NaN
180      NaN       15.0        NaN            NaN
190     25.0        NaN        NaN            NaN
199      NaN        NaN        NaN           10.0
200     35.0       20.0       55.0           15.0
240     40.0        NaN        NaN            NaN
250     45.0        NaN       60.0            NaN
257      NaN        NaN       65.0            NaN
260     50.0        NaN        NaN            NaN
270      NaN       25.0        NaN            NaN
280      NaN       30.0       70.0           20.0
299     55.0        NaN        NaN            NaN
300     60.0       35.0       80.0           30.0
340     65.0        NaN       85.0            NaN
350     75.0       40.0       90.0            NaN
360     85.0        NaN        NaN            NaN
370      NaN       50.0        NaN            NaN
380      NaN        NaN        NaN           40.0
388     90.0        NaN        NaN            NaN
399      NaN       55.0        NaN            NaN
400      NaN       75.0       95.0           45.0
410     95.0        NaN        NaN           50.0
420      NaN       80.0        NaN            NaN
422      NaN        NaN        NaN           55.0
430      NaN        NaN        NaN           60.0
433      NaN       85.0        NaN            NaN
450      NaN       90.0      100.0            NaN
460    100.0        NaN        NaN            NaN
480      NaN        NaN        NaN           65.0
490      NaN      100.0        NaN           70.0
499      NaN        NaN        NaN           75.0
500      NaN        NaN        NaN          100.0
'''

import matplotlib.pyplot as plt
#df2.plot()
#plt.show()


# 出现的图片是间断的， 因为有很多missing value ： NaN
# How to deal it? 把所有NaN fill with 0
#interpolate 可以让图看起来更平滑
df3 =df2.interpolate().fillna(0)


#reverse too cheap plot, to make the chart we want
df3['Too Cheap'] =100-df3['Too Cheap']
df3['Cheap'] =100-df3['Cheap']
df3.plot()
plt.show()

## finished
#下一步 interpret 这张图
